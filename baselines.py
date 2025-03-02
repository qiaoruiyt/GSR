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
from utils import AverageMeter, log_to_writer, EarlyStopper
from tqdm import tqdm
import gc
import time
from utils import evaluate_group_metrics
from data_utils import MySubset

class Trainer:
    """BaseTrainer 
    Args:
        args: arguments
        model: model to train
        train_dataset: training dataset
        val_dataset: validation dataset
        test_dataset: test dataset
        writer: wandb writer
    """
    def __init__(self, args, model, train_dataset, target_dataset, val_dataset, test_dataset=None, writer=None, ex_eval_fn=None, 
    early_stopper=None, verbose=False):
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
        self.epochs = args.epochs
        self.optimizer = args.optimizer
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.device = "cuda"
        self.early_stopper = early_stopper
        self.lasso = args.lasso

        if hasattr(train_dataset, 'n_groups'):
            self.n_groups = train_dataset.n_groups
        else:
            self.n_groups = None
            print("Warning: n_groups not found in train_dataset")

        if verbose:
            print("Initialized Trainer:")
            print(f"num_train: {len(train_dataset)}, num_target: {len(target_dataset)}, num_val: {len(val_dataset)}, num_test: {len(test_dataset)}")

    def train(self, sample_weights=None, early_stopper=None, resample=False):
        if early_stopper is not None:
            self.early_stopper = early_stopper
        num_epochs = self.epochs
        lr = self.inner_lr
        momentum = self.momentum
        batch_size = self.batch_size
        weight_decay = self.weight_decay
        lasso = self.lasso
        device = self.device
        model = self.model
        model = model.to(device)
        model.train()

        if resample:
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=sampler, num_workers=self.workers)
            sample_weights = None
        else:
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.workers)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers)
        if self.test_dataset is not None:
            test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers)

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=5000, tolerance_grad=1e-7, 
                tolerance_change=64 * torch.finfo(torch.float32).eps, 
                line_search_fn='strong_wolfe') # NOTE: learning rate has minimal effect when line_search_fn is used
                # NOTE: weighted decay can only be implemented using the closure function
        else:
            raise ValueError("Invalid optimizer")

        train_loss = None
        train_acc = None

        best_val_acc = -1
        best_train_acc = None
        best_train_loss = None
        best_epoch = None
        model_state_dict = None
        es_epoch = None

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
                if lasso > 0:
                    for param in model.parameters():
                        loss += lasso * torch.sum(torch.abs(param))
                loss.backward()
                return loss
            optimizer.step(closure)

            train_loss, train_acc = self.evaluate(model, train_loader)
        else:

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
                        loss = torch.sum(F.cross_entropy(output, target, reduction='none').flatten()*weights)
                    if lasso > 0:
                        for param in model.parameters():
                            loss += lasso * torch.sum(torch.abs(param))
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

        if self.n_groups is not None:
            val_loss, val_acc, val_group_loss, val_group_acc = evaluate_group_metrics(model, val_loader, device=self.device)
            test_loss, test_acc, test_group_loss, test_group_acc = evaluate_group_metrics(model, test_loader, device=self.device)
            best_val_wg = val_group_acc.min().item()
            train_info["best_val_wg"] = best_val_wg
            train_info["best_test_wg"] = test_group_acc.min().item()
            train_info["best_val"] = best_val_wg
        else:
            val_loss, val_acc = self.evaluate(model, val_loader)
            test_loss, test_acc = self.evaluate(model, test_loader)
            train_info["best_val"] = val_acc

        train_info["best_val_acc"] = val_acc
        train_info["best_val_loss"] = val_loss
        train_info["best_test_acc"] = test_acc
        train_info["best_test_loss"] = test_loss
        
        return train_info, model
        # return best_train_loss, best_train_acc, model

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
            for i in range(self.n_groups):
                self.writer.log({f"group_{i}_{header}loss": group_loss[i], f"group_{i}_{header}acc": group_acc[i]}, step=step)
            self.writer.log({f"worst_group_{header}loss": worst_group_loss, f"worst_group_{header}acc": worst_group_acc}, step=step)
        else:
            for i in range(self.n_groups):
                print(f"Group {i}: {header} loss: {group_loss[i]:.4f}, acc: {group_acc[i]:.4f}")
            print(f"Worst group: {header} loss: {worst_group_loss:.4f}, acc: {worst_group_acc:.4f}")

    def get_dataset_group_weights(self, dataset, sample_weights):
        num_train = sample_weights.size(0)
        n_groups = self.n_groups
        group_array = self.get_group_array(dataset)
        group_weights = torch.zeros(n_groups, device=self.device)
        for i in range(n_groups):
            group_indices = torch.arange(num_train)[group_array == i]
            if len(group_indices) == 0:
                continue
            group_weights[i] = sample_weights[group_indices].sum()
        return group_weights

    def get_group_array(self, dataset):
        group_array = []
        for datum in dataset:
            g = datum[2]
            group_array.append(g)
        return torch.tensor(group_array)

    def get_group_balanced_weights(self):
        num_train = len(self.train_dataset)
        n_groups = self.n_groups
        sample_weights = torch.ones(num_train, device=self.device)/n_groups
        group_array = self.get_group_array(self.train_dataset)
        group_weights = torch.zeros(n_groups, device=self.device).float()
        for i in range(n_groups):
            group_indices = torch.arange(num_train)[group_array == i]
            if len(group_indices) == 0:
                continue
            group_weights[i] = len(group_indices)
        sample_weights = sample_weights/group_weights[group_array]
        return sample_weights

def subsample_group_balanced_dataset(x_train, y_train, g_train):
    # create a new dataset with group balanced samples without reweighting
    n_groups = len(torch.unique(g_train))
    num_train = len(x_train)
    train_indices = torch.arange(num_train)
    g_idx = [train_indices[g_train == g] for g in range(n_groups)]
    min_g = min([len(g) for g in g_idx])
    for i in range(n_groups):
        g = g_idx[i]
        shuffled_idx = torch.randperm(len(g))
        g_idx[i] = g[shuffled_idx]
    train_indices = torch.cat([g[:min_g] for g in g_idx])
    print("New dataset length:", len(train_indices))
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    g_train = g_train[train_indices]
    return x_train, y_train, g_train



