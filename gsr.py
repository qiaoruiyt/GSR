import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import AverageMeter, log_to_writer, EarlyStopper, evaluate_group_metrics, get_scheduler, get_predictions
from tqdm import tqdm
import gc
import time
from torch_influence import LiSSAInfluenceModule, CGInfluenceModule, AutogradInfluenceModule, BaseInfluenceModule
from torch_influence import BaseObjective
from reweight_utils import FastExactInfluenceModule, GradOnlyInfluenceModule, FastHFInfluenceModule
from torch.linalg import cholesky


class MetaDataset(Dataset):
    '''Dataset with metadata that can be converted to groups using CombinatorialGrouper. 
    A meta(data) dataset usually has the following attributes:
    (data, target, group, confounder, index)
    ''' 
    def __init__(self, x, y, metadata):
        self.idx = torch.arange(len(x))
        self.x = x
        self.y = y
        self.metadata_array = metadata

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.metadata_array[index]

    def __len__(self):
        return len(self.x)


class SubpopDataset(Dataset):
    '''Dataset that works for subpopulation shift datasets. 
    A subpopulation shift dataset usually has the following attributes:
    (data, target, group, confounder)
    '''
    def __init__(self, x, y, group_array, metadata_array=None, confounder_array=None, all_group_indices=None):
        self.idx = torch.arange(len(x))
        self.x = x
        self.y = y
        self.group_array = group_array
        self.all_group_indices = all_group_indices
        self.metadata_array = torch.zeros(len(x)) if metadata_array is None else metadata_array
        self.confounder_array = torch.zeros(len(x)) if confounder_array is None else confounder_array
        if all_group_indices is not None:
            self.n_groups = len(all_group_indices)
        else:
            self.n_groups = len(torch.unique(group_array))

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.group_array[index], self.metadata_array[index]

    def __len__(self):
        return len(self.x)


class IndexedDataset(Dataset):
    '''Indexed Dataset Wrapper 
    An indexed subpopulation shift dataset usually has the following attributes:
    (data, target, group, confounder, index)
    An indexed meta(data) dataset usually has the following attributes:
    (data, target, metadata, index)
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.idx = torch.arange(len(dataset))
        if hasattr(dataset, 'n_groups'):
            self.n_groups = dataset.n_groups
            self.group_array = dataset.group_array
            self.all_group_indices = dataset.all_group_indices
        else:
            self.n_groups = None
        self.idx_ref = torch.arange(len(self.idx))

    def __getitem__(self, index):
        return self.dataset[self.idx[index]] + (self.idx_ref[index],)

    def __len__(self):
        return len(self.idx)


class NoninvInfluenceModule(BaseInfluenceModule):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            check_eigvals: bool = False
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        d = flat_params.shape[0]
        hess = 0.0

        for batch, batch_size in self._loader_wrapper(train=True):
            def f(theta_):
                self._model_reinsert_params(self._reshape_like_params(theta_))
                return self.objective.train_loss(self.model, theta_, batch)

            hess_batch = torch.autograd.functional.hessian(f, flat_params).detach()
            hess = hess + hess_batch * batch_size

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
            hess = hess / len(self.train_loader.dataset)
            if damp > 0:
                hess = hess + damp * torch.eye(d, device=hess.device)

            if check_eigvals:
                eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
                logging.info("hessian min eigval %f", np.min(eigvals).item())
                logging.info("hessian max eigval %f", np.max(eigvals).item())
                if not bool(np.all(eigvals >= 0)):
                    raise ValueError()

        self.hess = hess

    def inverse_hvp(self, vec):
        if self.damp > 0:
            L = torch.linalg.cholesky(self.hess, upper=False)
            ihvp = torch.cholesky_solve(vec.unsqueeze(1), L, upper=False).squeeze(1)
        else:
            ihvp = torch.linalg.solve(self.hess, vec)
        return ihvp


class WeightedObjective(BaseObjective):
    def __init__(self, sample_weights=None, weight_decay=0, normalize=False):
        self.sample_weights = sample_weights
        self.weight_decay = weight_decay
        self.normalize = normalize
        # if self.normalize:
        #     self.sample_weights = self.sample_weights / self.sample_weights.sum()

    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        labels = batch[1] 
        indices = batch[-1]
        weights = self.sample_weights[indices]
        # should not divide by the sum of weights here because influence uses batch-size 1 for per-sample grad
        if self.sample_weights is None:
            loss = F.cross_entropy(outputs, labels)
        else:
            loss = torch.sum(F.cross_entropy(outputs, labels, reduction='none') * weights)
        return loss

    def unweighted_train_loss(self, model, params, batch):
        outputs = model(batch[0])
        return F.cross_entropy(outputs, batch[1]) + self.train_regularization(params)

    def train_regularization(self, params):
        if self.weight_decay == 0:
            return 0
        return self.weight_decay * torch.square(params.norm())

    # training loss by default taken to be 
    # train_loss_on_outputs + train_regularization

    def test_loss(self, model, params, batch):
        outputs = model(batch[0])
        return F.cross_entropy(outputs, batch[1])


class GSRTrainer:
    """Trainer for Group-robust Sample Reweighting using Influence Functions
    Args:
        args: arguments
        model: model to train
        train_dataset: training dataset
        val_dataset: validation dataset
        test_dataset: test dataset
        writer: wandb writer
        pgd: whether to use projected gradient descent for sample weight updates
        label_flip: whether to flip the labels for negative sample weights, only used when pgd is False
    """
    def __init__(self, args, model, train_dataset, target_dataset, val_dataset, test_dataset=None, writer=None, strategy='inf', dro_metric='loss', ex_eval_fn=None, 
    pgd=False, label_correction=False, early_stopper=None, device='cuda'):
        print(args)
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.target_dataset = target_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.writer = writer
        self.ex_eval_fn = ex_eval_fn

        self.inner_lr = args.inner_lr
        self.outer_lr = args.outer_lr
        self.outer_lr_scheduler = None
        if args.outer_lr_scheduler is not None:
            self.outer_lr_scheduler = get_scheduler(args.outer_lr_scheduler)
        self.outer_max_magnitude = args.outer_max_magnitude
        self.normalize_weights_batch = args.normalize_weights_batch # normalize weights in each batch
        self.normalize_weights = args.normalize_weights # normalize weights after each outer iteration
        self.outer_grad_clip = args.outer_grad_clip

        self.epochs = args.epochs
        self.max_outer_iter = args.max_outer_iter
        self.optimizer = args.optimizer
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum

        self.temperature = args.temperature
        self.soft_update = not args.hard_update
        self.strategy = strategy
        self.dro_metric = args.dro_metric
        self.pgd = args.pgd
        self.resample = args.resample
        self.early_stopper = early_stopper
        self.label_correction = args.label_correction 
        if strategy == 'inf':
            self.inf = args.inf
            self.damp = args.damp
        else:
            raise ValueError("Invalid strategy")
        self.batch_size = args.batch_size
        self.workers = args.workers

        self.device = device

        if hasattr(target_dataset, 'n_groups'):
            self.n_groups = target_dataset.n_groups
        else:
            self.n_groups = None

        self.multiplicative_updates = args.multiplicative_updates if hasattr(args, 'multiplicative_updates') else False 
        self.group_score_weights = torch.ones(self.n_groups, device=self.device)/self.n_groups

        print("Initialized GSRTrainer:")
        print(f"num_train: {len(train_dataset)}, num_target: {len(target_dataset)}, num_val: {len(val_dataset)}, num_test: {len(test_dataset)}")

    def train(self, model, train_loader, val_loader=None, sample_weights=None, outer_iter=None, early_stopper=None):
        if early_stopper is not None:
            self.early_stopper = early_stopper
        num_epochs = self.epochs
        lr = self.inner_lr
        momentum = self.momentum
        weight_decay = self.weight_decay
        device = self.device
        model = model.to(device)
        model.train()
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=300, tolerance_grad=1e-7, 
                tolerance_change=64 * torch.finfo(torch.float32).eps, 
                line_search_fn='strong_wolfe') # NOTE: learning rate has minimal effect when line_search_fn is used
                # NOTE: weighted decay can only be implemented using the closure function
        else:
            raise ValueError("Invalid optimizer")

        if self.optimizer == 'lbfgs': # Does not do mini-batch training
            train_dataset = train_loader.dataset
            data = train_dataset.dataset.x.to(device)
            target = train_dataset.dataset.y.to(device)
            sample_weights = sample_weights.to(device)
            def closure():
                optimizer.zero_grad()
                output = model(data)

                if sample_weights is None:
                    loss = F.cross_entropy(output, target)
                else:
                    loss = torch.sum(F.cross_entropy(output, target, reduction='none').flatten()*sample_weights/sample_weights.sum())
                if weight_decay > 0:
                    for param in model.parameters():
                        loss += weight_decay * torch.sum(torch.square(param))
                loss.backward()
                return loss
            optimizer.step(closure)

            train_loss, train_acc = self.evaluate(model, train_loader)

            train_info = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "best_train_loss": train_loss,
                "best_train_acc": train_acc,
                "best_epoch": None,
                "es_epoch": None,
            }
            return train_info, model

        train_loss = None
        train_acc = None

        best_val_acc = -1
        best_train_acc = None
        best_train_loss = None
        best_epoch = None
        model_state_dict = None
        es_epoch = None

        for i in tqdm(range(num_epochs), desc="inner loop"):
            if self.early_stopper is not None:
                self.early_stopper.reset()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            for data, target, _, _, indices in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)

                if sample_weights is None:
                    loss = F.cross_entropy(output, target)
                else:
                    weights = sample_weights[indices]
                    # if self.normalize_weights_batch:
                    #     loss = torch.sum(F.cross_entropy(output, target, reduction='none').flatten()*weights/weights.sum())
                    # else:
                    loss = torch.sum(F.cross_entropy(output, target, reduction='none').flatten()*weights)
                loss.backward()
                optimizer.step()
                
                acc = (output.argmax(1) == target).float().mean()
                acc_meter.update(acc.item(), data.size(0))
                loss_meter.update(loss.item(), data.size(0))

            train_loss = loss_meter.avg
            train_acc = acc_meter.avg

            # evaluate on the validation set
            if val_loader is not None:
                if self.n_groups is not None:
                    val_loss, val_acc, val_group_loss, val_group_acc = evaluate_group_metrics(model, val_loader, device=self.device)
                    val_acc = val_group_acc.min().item() # assign as the worst_group_acc
                else:
                    val_loss, val_acc = self.evaluate(model, val_loader)
                    
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_train_acc = train_acc
                    best_train_loss = train_loss
                    best_epoch = i
                    model_state_dict = deepcopy(model.state_dict())

                if self.early_stopper is not None:
                    if self.early_stopper.step(val_loss):
                        es_epoch = i
                        break
            elif i == num_epochs - 1:
                model_state_dict = deepcopy(model.state_dict())
                best_train_acc = train_acc
                best_train_loss = train_loss

        # load the best model
        model.load_state_dict(model_state_dict)

        # training info
        train_info = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "best_train_loss": best_train_loss,
            "best_train_acc": best_train_acc,
            "best_epoch": best_epoch,
            "es_epoch": es_epoch,
        }
        
        return train_info, model

    def evaluate(self, model, loader):
        device = self.device
        model.eval()
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for data, target, _, _, _, in loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)

                acc = (output.argmax(1) == target).float().mean()
                acc_meter.update(acc.item(), data.size(0))

                loss = F.cross_entropy(output, target)
                loss_meter.update(loss.item(), data.size(0))
        return loss_meter.avg, acc_meter.avg

    def log_group_metrics(self, group_loss, group_acc, header='', step=None):
        worst_group_acc = group_acc.min().item()
        worst_group_loss = group_loss[group_acc.argmin()].item()
        if self.writer is not None:
            if header != '':
                header = header + "_"
            for i in range(len(group_loss)):
                self.writer.log({f"group_{i}_{header}loss": group_loss[i], f"group_{i}_{header}acc": group_acc[i]}, step=step)
            self.writer.log({f"worst_group_{header}loss": worst_group_loss, f"worst_group_{header}acc": worst_group_acc}, step=step)
        else:
            for i in range(len(group_loss)):
                print(f"Group {i}: {header} loss: {group_loss[i]:.4f}, acc: {group_acc[i]:.4f}")
            print(f"Worst group: {header} loss: {worst_group_loss:.4f}, acc: {worst_group_acc:.4f}")

    # NOTE: not used
    def get_grad_on_dataset(self, model, dataset, wg=True):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers)
        
        device = self.device
        model = model.to(device)
        model.train()
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        group_acc = torch.zeros(self.n_groups, device=device)
        group_loss = torch.zeros(self.n_groups, device=device)
        group_counts = torch.zeros(self.n_groups, device=device)
        group_grads = []
        for data, target, group, confounder, index in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            acc = (output.argmax(1) == target).float().mean()
            acc_meter.update(acc.item(), data.size(0))

            loss = F.cross_entropy(output, target)
            loss_meter.update(loss.item(), data.size(0))

            for i in range(self.n_groups):
                group_indices = group == i
                group_counts[i] += group_indices.sum()
                group_acc[i] += (output[group_indices].argmax(1) == target[group_indices]).float().sum()
                group_loss[i] += F.cross_entropy(output[group_indices], target[group_indices], reduction='sum')
        group_acc = group_acc / group_counts
        group_loss = group_loss / group_counts
        # calculate group grads
        for i in range(self.n_groups):
            if wg: # only consider the worst group
                if group_acc[i] != group_acc.min():
                    continue
            grad = torch.autograd.grad(group_loss[i], model.parameters())
            group_grads.append(grad)
        return loss_meter.avg, acc_meter.avg, group_loss, group_acc

    def get_weight_update(self, model, train_dataset, target_dataset, sample_weights=None, group_loss=None):
        if self.strategy == 'inf':
            if self.args.inf_batch_size is not None:
                batch_size = self.args.inf_batch_size
            else:
                if self.args.heldout_path is None: # usually this indicates full retraining
                    batch_size = self.batch_size // 4 # a heuristic to reduce memory usage for influence function
                else: # this is last-layer retraining
                    batch_size = self.batch_size * 8
        else:
            batch_size = self.batch_size
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers)
        # target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers)
        # NOTE: For LLR, we use the full dataset
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        target_loader = DataLoader(target_dataset, batch_size=len(target_dataset), shuffle=False)
        num_train = len(train_loader.dataset)
        num_target = len(target_loader.dataset)
        train_indices = torch.arange(num_train)
        val_indices = torch.arange(num_target)

        objective = WeightedObjective(sample_weights, self.weight_decay)
        if self.inf == 'cg':
            module = CGInfluenceModule(
                model=model,
                objective=objective,
                train_loader=train_loader,
                test_loader=target_loader,
                device=self.device,
                damp=self.damp, 
            )
        elif self.inf == 'lissa':
            module = LiSSAInfluenceModule(
                model=model,
                objective=objective,
                train_loader=train_loader,
                test_loader=target_loader,
                device=self.device,
                damp=self.damp, 
                repeat=1, 
                depth=self.args.lissa_depth, 
                scale=1e4,
            )
        elif self.inf == 'exact':
            # module = AutogradInfluenceModule(
            module = FastExactInfluenceModule(
                model=model,
                objective=objective,
                train_loader=train_loader,
                test_loader=target_loader,
                device=self.device,
                damp=self.damp,
            )
        elif self.inf == 'noninv':
            module = NoninvInfluenceModule(
                model=model,
                objective=objective,
                train_loader=train_loader,
                test_loader=target_loader,
                device=self.device,
                damp=self.damp,
            )
        elif self.inf == 'last':
            # module = GradOnlyInfluenceModule(
            module = FastHFInfluenceModule(
                model=model,
                objective=objective,
                train_loader=train_loader,
                test_loader=target_loader,
                device=self.device,
            )
        elif self.inf == 'datainf':
            raise NotImplementedError
        else:
            raise ValueError("Invalid influence function")

        if group_loss is not None:
            group_scores = torch.zeros(self.n_groups, num_train, device=self.device)
            for i in range(self.n_groups):
                if target_loader.dataset.all_group_indices is not None:
                    target_group_indices = target_loader.dataset.all_group_indices[i]
                else:
                    target_group_indices = torch.arange(num_target)[target_loader.dataset.group_array == i]
                if len(target_group_indices) == 0:
                    continue
                group_scores[i] = module.influences(train_indices, target_group_indices)
            
            if self.multiplicative_updates:
                updates = torch.exp(group_loss)/self.temperature 
                self.group_score_weights = self.group_score_weights * updates
                self.group_score_weights = self.group_score_weights / self.group_score_weights.sum()
                group_scores = group_scores * self.group_score_weights.view(-1, 1)
                scores = group_scores.sum(dim=0)
            elif self.soft_update:
                group_score_weights = F.softmax(group_loss/self.temperature, dim=0).view(-1, 1)
                group_scores = group_scores * group_score_weights
                scores = group_scores.sum(dim=0)
            else: # only use the worst group scores
                worst_group = group_loss.argmax()
                scores = group_scores[worst_group] # NOTE: This hurts convergence
        else:
            scores = module.influences(train_indices, val_indices)
            group_scores = None

        return scores, group_scores

    def get_train_group_weights(self, train_loader, sample_weights):
        num_train = len(train_loader.dataset)
        n_groups = train_loader.dataset.n_groups
        group_weights = torch.zeros(n_groups, device=self.device)
        for i in range(n_groups):
            if train_loader.dataset.all_group_indices is not None:
                group_indices = train_loader.dataset.all_group_indices[i]
            else:
                group_indices = torch.arange(num_train)[train_loader.dataset.group_array == i]
            if len(group_indices) == 0:
                continue
            group_weights[i] = sample_weights[group_indices].sum()
        return group_weights

    def solve(self):
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        target_dataset = self.target_dataset
        model = self.model

        num_train = len(train_dataset)

        writer = self.writer

        outer_lr = self.outer_lr
        outer_max_magnitude = self.outer_max_magnitude
        batch_size = self.batch_size

        # initialize sample weights
        sample_weights = torch.full([num_train], 1, dtype=torch.float, device=self.device)
        if self.normalize_weights:
            sample_weights = sample_weights / num_train

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers, pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers, pin_memory=True)

        eval_test = False
        if self.test_dataset is not None:
            eval_test = True
            test_dataset = self.test_dataset
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers, pin_memory=True)
        pin_train_mem = not self.resample
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.workers, pin_memory=pin_train_mem)

        if writer is not None:
            group_weights = self.get_train_group_weights(train_loader, sample_weights)
            for i in range(self.n_groups):
                writer.log({"group_weights_{}".format(i): group_weights[i]}, step=0)

        best_val_acc = 0 # model selection using the validation set
        best_val_wg = 0
        best_iter = 0
        best_train = 0
        best_test = 0
        best_test_wg = 0
        best_sample_weights = sample_weights
        best_model = model

        time_taken = 0
        inner_time_meter = AverageMeter()
        outer_time_meter = AverageMeter()

        for outer_iter in range(1, self.max_outer_iter+1):
            time_taken_inner = 0
            time_taken_outer = 0
            inner_start_time = time.time()

            model_copy = deepcopy(model)
            # train on the training/heldout set, select the best model based on the validation set
            if self.resample:
                sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_train, replacement=True)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=self.workers)
                train_info, model_copy = self.train(model_copy, train_loader, val_loader)
            else:
                train_info, model_copy = self.train(model_copy, train_loader, val_loader, sample_weights)
            train_loss, train_acc = train_info['best_train_loss'], train_info['best_train_acc']
            best_epoch = train_info['best_epoch']

            inner_end_time = time.time()
            time_taken_inner = inner_end_time - inner_start_time
            time_taken += time_taken_inner

            outer_start_time = time.time()
            if self.n_groups is not None:
                val_loss, val_acc, val_group_loss, val_group_acc = evaluate_group_metrics(model_copy, val_loader, device=self.device)
                self.log_group_metrics(val_group_loss, val_group_acc, step=outer_iter, header='val')
                target_loss, target_acc, target_group_loss, target_group_acc = evaluate_group_metrics(model_copy, target_loader, device=self.device)
                self.log_group_metrics(target_group_loss, target_group_acc, step=outer_iter, header='target')
                if eval_test:
                    test_loss, test_acc, test_group_loss, test_group_acc = evaluate_group_metrics(model_copy, test_loader, device=self.device)
                    self.log_group_metrics(test_group_loss, test_group_acc, step=outer_iter, header='test')

                wg_val_acc = val_group_acc.min().item()
                wg_test_acc = test_group_acc.min().item()
                # model selection
                if wg_val_acc > best_val_wg:
                    best_val_acc = val_acc
                    best_val_wg = wg_val_acc
                    best_iter = outer_iter
                    best_train = train_acc
                    best_test = test_acc
                    best_test_wg = wg_test_acc
                    best_sample_weights = deepcopy(sample_weights)
                    best_model = deepcopy(model_copy)
            else:
                val_loss, val_acc = self.evaluate(model_copy, val_loader)
                if eval_test:
                    test_loss, test_acc = self.evaluate(model_copy, test_loader)

                # model selection
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_iter = outer_iter
                    best_train = train_acc
                    best_test = test_acc
                    best_sample_weights = deepcopy(sample_weights)
                    best_model = deepcopy(model_copy)
            
            print(f"step {outer_iter}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")
            if writer is not None:
                writer.log({"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc}, step=outer_iter)
                writer.log({"sample_weights_max": sample_weights.max().item(), "sample_weights_min": sample_weights.min().item()}, step=outer_iter)
                writer.log({"time_taken": time_taken}, step=outer_iter)
                writer.log({"best_epoch": best_epoch}, step=outer_iter)

            if outer_iter < self.max_outer_iter:
                # calculate the influence score of the training set on the validation set
                if self.n_groups is not None:
                    if self.dro_metric == 'loss':
                        group_loss = val_group_loss
                    elif self.dro_metric == 'acc':
                        group_loss = 1 - val_group_acc
                    scores, group_scores = self.get_weight_update(model_copy, train_dataset, target_dataset, sample_weights, group_loss=group_loss)
                else:
                    scores, _ = self.get_weight_update(model_copy, train_dataset, target_dataset, sample_weights)

                if torch.isnan(scores).any():
                    raise ValueError("NaN in calculating influence scores")

                # track score status
                if writer is not None:
                    writer.log({"raw_scores_max": scores.max().item(), "raw_scores_min": scores.min().item(), "raw_scores_mean": scores.mean().item()}, step=outer_iter)
                    writer.log({"raw_scores_norm": torch.linalg.vector_norm(scores).item()}, step=outer_iter)

                if self.outer_max_magnitude is not None and self.outer_max_magnitude > 0:
                    scores_mag = torch.abs(scores)
                    scores = scores / scores_mag.max() * outer_max_magnitude 
                
                if self.outer_grad_clip is not None and self.outer_grad_clip > 0:
                    score_norm = torch.linalg.vector_norm(scores)
                    clip_coef = self.outer_grad_clip / (score_norm + 1e-9)
                    clip_coef = torch.clamp(clip_coef, max=1)
                    scores = scores * clip_coef
                scores = scores.to(self.device)
                if self.outer_lr_scheduler is not None:
                    lr_factor = self.outer_lr_scheduler.step()
                    outer_lr = outer_lr * lr_factor
                sample_weights += outer_lr * scores
                if self.pgd:
                    sample_weights = torch.clamp(sample_weights, 0)
                if self.normalize_weights:
                    sample_weights = sample_weights / sample_weights.sum()

                outer_end_time = time.time()
                time_taken_outer = outer_end_time - outer_start_time
                time_taken += time_taken_outer
                inner_time_meter.update(time_taken_inner)
                outer_time_meter.update(time_taken_outer)

                if writer is not None:
                    group_weights = self.get_train_group_weights(train_loader, sample_weights)
                    for i in range(self.n_groups):
                        writer.log({"group_weights_{}".format(i): group_weights[i]}, step=outer_iter)

                    writer.log({"time_taken_inner": time_taken_inner, "time_taken_outer": time_taken_outer}, step=outer_iter)

            # clean up
            del model_copy
            torch.cuda.empty_cache()
            gc.collect()

        results = {
            "best_val_acc": best_val_acc,
            "best_val_wg": best_val_wg,
            "best_train_acc": best_train,
            "best_test_acc": best_test,
            "best_test_wg": best_test_wg,
            "best_iter": best_iter,
        }

        print(f"total time taken: {time_taken:.2f}s")
        print(f"average inner time: {inner_time_meter.avg:.2f}s")
        print(f"average outer time: {outer_time_meter.avg:.2f}s")
        if writer is not None:
            writer.log({"total_time": time_taken, "average_inner_time": inner_time_meter.avg, "average_outer_time": outer_time_meter.avg})
            writer.log({"best_val_acc": best_val_acc, "best_val_wg": best_val_wg, "best_train_acc": best_train, "best_test_acc": best_test, "best_test_wg": best_test_wg})

        return results, best_sample_weights, best_model