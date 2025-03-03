import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import random
import argparse
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from copy import deepcopy
import torchvision
from wilds.common.grouper import CombinatorialGrouper
from models import DistilBertClassifier, BertClassifier


def add_label_noise(y, noise_level):
    y = y.clone()
    classes = torch.unique(y)
    n_classes = len(classes)
    n_noise = int(noise_level * len(y))
    noise_indices = np.random.choice(len(y), n_noise, replace=False)
    current_labels = y[noise_indices]
    # change the current label to a different class
    new_labels = torch.randint(0, n_classes, (n_noise,))
    for i in range(n_noise):
        while new_labels[i] == current_labels[i]:
            new_labels[i] = torch.randint(0, n_classes, (1,))
    y[noise_indices] = new_labels
    return y, noise_indices


def evaluate_group_metrics(model, loader, device='cuda'):
    n_groups = loader.dataset.n_groups
    use_prepared_group_indices = False
    if loader.dataset.all_group_indices is not None:
        assert n_groups == len(loader.dataset.all_group_indices)
        use_prepared_group_indices = True
    model.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    group_acc = torch.zeros(n_groups, device=device)
    group_loss = torch.zeros(n_groups, device=device)
    group_counts = torch.zeros(n_groups, device=device)
    with torch.no_grad():
        for data, target, group, confounder, index in loader:
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)
            acc = (pred == target).float().mean()
            acc_meter.update(acc.item(), data.size(0))

            loss = F.cross_entropy(logits, target)
            loss_meter.update(loss.item(), data.size(0))

            for i in range(n_groups):
                if use_prepared_group_indices:
                    group_indices = torch.isin(index, loader.dataset.all_group_indices[i])
                else:
                    group_indices = group == i
                group_counts[i] += group_indices.sum()
                group_acc[i] += (logits[group_indices].argmax(1) == target[group_indices]).float().sum()
                group_loss[i] += F.cross_entropy(logits[group_indices], target[group_indices], reduction='sum')
    group_acc = group_acc / group_counts
    group_loss = group_loss / group_counts
    return loss_meter.avg, acc_meter.avg, group_loss, group_acc


def get_scheduler(lr_scheduler_name, args=None):
    if lr_scheduler_name == 'step':
        return StepLR()
    else:
        raise ValueError(f"Unknown scheduler: {lr_scheduler_name}")


class StepLR:
    def __init__(self, step_size=30, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma
        self.steps = -1

    def step(self):
        self.steps += 1
        if self.steps > 0 and self.steps % self.step_size == 0:
            return self.gamma
        return 1


class EarlyStopper:
    def __init__(self, patience=5, tolerance=1e-4, min_steps=30):
        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.min_steps = min_steps
        self.steps = 0

    def step(self, validation_loss):
        self.steps += 1
        if self.steps < self.min_steps:
            return False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.tolerance):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.steps = 0


class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def maybe_dictionarize_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) == 2:
        return {'x': batch[0], 'y': batch[1]}
    elif len(batch) == 3:
        return {'x': batch[0], 'y': batch[1], 'g': batch[2]}
    elif len(batch) == 4:
        return {'x': batch[0], 'y': batch[1], 'g': batch[2], 'c': batch[3]}
    elif len(batch) == 5:
        return {'x': batch[0], 'y': batch[1], 'g': batch[2], 'c': batch[3], 'idx': batch[4]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_y_p(g, n_places):
    y = g // n_places
    p = g % n_places
    return y, p


def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)


def write_dict_to_tb(writer, dict, prefix, step):
    for key, value in dict.items():
        writer.add_scalar(f"{prefix}{key}", value, step)


def log_to_writer(logger, key, value, step=None):
    # if logger is tensorboard logger
    if hasattr(logger, 'add_scalar'):
        logger.add_scalar(f"{key}", value, step)
    # if logger is wandb logger
    elif hasattr(logger, 'log'):
        logger.log({f"{key}": value}, step=step)
    else:
        raise ValueError("Unknown logger")


def get_predictions(model, dataset, device='cuda', num_workers=0):
    model = model.to(device)
    model.eval()
    y_preds = []
    y_true = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=num_workers)
    with torch.no_grad():
        for batch in loader:
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data)
            y_preds.append(output.argmax(1).cpu())
            y_true.append(target.cpu())
    y_preds = torch.cat(y_preds)
    y_true = torch.cat(y_true)
    return y_preds, y_true


def get_results(acc_groups, get_yp_func):
    groups = acc_groups.keys()
    results = {
            f"accuracy_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}": acc_groups[g].avg
            for g in groups
    }
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"mean_accuracy" : all_correct / all_total})
    results.update({"worst_accuracy" : min(results.values())})
    return results


def evaluate(model, loader, get_yp_func, multitask=False, predict_place=False):
    model.eval()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(loader.dataset.n_groups)}
    if multitask:
        acc_place_groups = {g_idx: AverageMeter() for g_idx in range(trainset.n_groups)}

    with torch.no_grad():
        for x, y, g, p in tqdm.tqdm(loader):
            x, y, p = x.cuda(), y.cuda(), p.cuda()
            if predict_place:
                y = p

            logits = model(x)
            if multitask:
                logits, logits_place = logits
                update_dict(acc_place_groups, p, g, logits_place)

            update_dict(acc_groups, y, g, logits)
    model.train()
    if multitask:
        return get_results(acc_groups, get_yp_func), get_results(acc_place_groups, get_yp_func)
    return get_results(acc_groups, get_yp_func)


class MultiTaskHead(nn.Module):
    def __init__(self, n_features, n_classes_list):
        super(MultiTaskHead, self).__init__()
        self.fc_list = [
            nn.Linear(n_features, n_classes).cuda()
            for n_classes in n_classes_list
        ]

    def forward(self, x):
        outputs = []
        for head in self.fc_list:
            out = head(x)
            outputs.append(out)
        return outputs


class MySubset(torch.utils.data.Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)


def get_training_args():
    parser = argparse.ArgumentParser(description="Training the feature extractor")
    parser = add_base_args(parser)

    parser.add_argument("--pretrained_model", action='store_true', help="Use pretrained model")
    parser.add_argument("--scheduler", action='store_true', help="Learning rate scheduler")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--momentum_decay", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--held_out_ratio", type=float, default=0.1)
    args = parser.parse_args()
    args.root_dir = os.path.expanduser(args.root_dir)
    args.data_dir = os.path.expanduser(args.data_dir)
    args.ckpt_path = os.path.expanduser(args.ckpt_path)
    args.output_dir = os.path.expanduser(args.output_dir)
    return args


def get_last_layer_retraining_args():
    parser = argparse.ArgumentParser(description="GSR + LLR")
    parser = add_base_args(parser)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--inf_batch_size', default=None, type=int)
    parser.add_argument('--inf', default='exact', choices=['lissa', 'cg', 'exact', 'datainf', 'last', 'noninv'], type=str)
    parser.add_argument('--preprocess_bs', default=128, type=int)
    parser.add_argument('--preprocess_workers', default=4, type=int)
    parser.add_argument('--heldout_path', default=None, type=str)
    parser.add_argument('--val_subsample', default=0.5, type=float, help="The fraction of the validation set to use, and the rest is used as target set")
    parser.add_argument('--target_fraction', default=None, type=float, help="Fraction of the target set to use")
    parser.add_argument('--val_fraction', default=None, type=float, help="Fraction of the validation set to use")
    parser.add_argument('--no_target', default=False, action='store_true')
    parser.add_argument('--hard_update', default=False, action='store_true')
    parser.add_argument('--fast_solve', default=False, action='store_true')
    parser.add_argument('--dro_metric', default='acc', choices=['acc', 'loss'], type=str)
    parser.add_argument('--pgd', default=True, action='store_true')
    parser.add_argument('--label_correction', default=False, action='store_true')
    parser.add_argument('--damp', default=0.0, type=float)
    parser.add_argument('--base_dir', default='output', type=str)
    parser.add_argument('--group_dro', default=False, action='store_true')
    parser.add_argument('--group_balanced', default=False, action='store_true')
    parser.add_argument('--gb_subsample', default=False, action='store_true')
    parser.add_argument('--resample', default=False, action='store_true')
    parser.add_argument('--train_val', default=False, action='store_true')
    parser.add_argument('--lasso', default=0, type=float)
    parser.add_argument('--eval_base', default=False, action='store_true')
    parser.add_argument('--no_cache', default=False, action='store_true')
    parser.add_argument('--outer_grad_clip', default=None, type=float)
    parser.add_argument('--class_balanced', default=False, action='store_true')
    parser.add_argument('--mis_balanced', default=False, action='store_true')
    parser.add_argument('--keep_val_group', default=False, action='store_true')
    parser.add_argument('--outer_lr_scheduler', default=None, type=str)
    parser.add_argument('--save_stats', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--label_noise', default=None, type=float)
    parser.add_argument('--group_noise', default=None, type=float)
    parser.add_argument('--target_noise', default=None, type=float)
    parser.add_argument('--no_heldout', default=False, action='store_true')
    parser.add_argument('--multiple_groupers', default=False, action='store_true')
    parser.add_argument('--filter_groups', default=False, action='store_true')
    parser.add_argument('--use_metadata', default=False, action='store_true')
    parser.add_argument('--multiplicative_updates', default=True, action='store_true')

    args = parser.parse_args()
    args.root_dir = os.path.expanduser(args.root_dir)
    args.data_dir = os.path.expanduser(args.data_dir)
    args.ckpt_path = os.path.expanduser(args.ckpt_path)
    args.output_dir = os.path.expanduser(args.output_dir)
    return args


def get_full_reweighting_args():
    parser = argparse.ArgumentParser(description="GSR")
    parser = add_base_args(parser)

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.ckpt_path = os.path.expanduser(args.ckpt_path)
    return args


def add_base_args(parser):
    parser.add_argument("--root_dir", type=str, default='~/data', help="Dataset root directory")
    parser.add_argument("--data_dir", type=str, default='~/data/waterbirds_v1.0', help="Train dataset directory")
    parser.add_argument("--output_dir", type=str, default="logs/", help="Path to save results")
    parser.add_argument("--dataset", type=str, default='waterbirds', help="Dataset name")
    parser.add_argument("--augment_data", action='store_true', help="Train data augmentation")
    parser.add_argument("--download", action='store_true', help="Download dataset")
    parser.add_argument("--model", type=str, default='resnet50', help="Model name")
    parser.add_argument("--ckpt_path", type=str, default='output/final_checkpoint.pt', help="Checkpoint path")
    parser.add_argument('--seed', type=int, default=1, metavar='seed', help='random seed (default: 1)') # seed 1 was used by DFR and AFR
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--outer_lr', default=1, type=float)
    parser.add_argument('--outer_max_magnitude', default=0, type=float)
    parser.add_argument('--inner_lr', default=0.1, type=float)
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'lbfgs'], type=str)
    parser.add_argument('--max_outer_iter', default=10, type=int)
    parser.add_argument('--runs_name', default="gsr-test", type=str)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--lissa_depth', default=100, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--no_logs', action='store_true')
    parser.add_argument('--normalize_weights_batch', default=False, action='store_true')
    parser.add_argument('--normalize_weights', default=False, action='store_true')
    parser.add_argument('--strategy', default='inf', choices=['inf', 'tracin', 'ntk'], type=str)
    parser.add_argument('--no_wandb', default=False, action='store_true')
    parser.add_argument("--multitask", action='store_true', help="Predict label and group")
    parser.add_argument("--max_token_length", type=int, default=300)
    return parser


def get_model(model_name, n_classes):
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        d = model.fc.in_features
        model.fc = torch.nn.Linear(d, n_classes)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        d = model.fc.in_features
        model.fc = torch.nn.Linear(d, n_classes)
    elif 'distilbert-' in model_name:
        # by default the model only output the logits
        # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
        model = DistilBertClassifier(model_name, n_classes)
    elif 'bert-' in model_name:
        model = BertClassifier(model_name, n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def get_optimizer(model_parameters, args):
    optimizer_name = args.optimizer
    if optimizer_name == 'sgd':
        return torch.optim.SGD(model_parameters, lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)
    elif optimizer_name == 'adam':
        print("Using AdamW instead of Adam by default")
        return torch.optim.AdamW(model_parameters, lr=args.init_lr, weight_decay=args.weight_decay)
    elif optimizer_name == 'lbfgs':
        raise ValueError("L-BFGS should only be used for last-layer retraining instead of representation learning")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_feature_extractor(model):
    if 'bert' in model.__class__.__name__.lower():
        model.classifier = nn.Identity()
    else:
        model.fc = nn.Identity()
    return model

class OneGrouper:
    def __init__(self, groupers, sample_groups=True):
        self.groupers = groupers
        if hasattr(groupers, 'n_groups'):
            self.n_groups = groupers.n_groups
        else:
            self.n_groups = 1
        self.sample_groups = sample_groups

    def metadata_to_group_indices(self, metadata):
        if isinstance(self.groupers, list):
            all_group_indices = []
            for i in range(len(self.groupers)): # groupers
                grouper = self.groupers[i]
                groups = grouper.metadata_to_group(metadata)
                id_ones = []
                id_zeros = []
                for j in range(len(groups)): # instances
                    group = groups[j]
                    identity_var = grouper.groupby_fields[0]
                    group_str = grouper.group_field_str(group)
                    # detected the mention of an identity
                    if f'{identity_var}:1' in group_str:
                        if f'y:0' in group_str:
                            id_zeros.append(j) # appendix the index
                        else:
                            id_ones.append(j)
                id_zeros = torch.tensor(id_zeros)
                id_ones = torch.tensor(id_ones)
                all_group_indices.append(id_zeros)
                all_group_indices.append(id_ones)
            return all_group_indices

    def sanity_check(self, metadata, all_group_indices):
        for i in range(len(self.groupers)):
            grouper = self.groupers[i]
            groups = grouper.metadata_to_group(metadata)
            group_indices_0 = all_group_indices[int(2*i)]
            group_indices_1 = all_group_indices[int(2*i+1)]
            for j in group_indices_0 + group_indices_1:
                group = groups[j]
                group_str = grouper.group_field_str(group)
                identity_var = grouper.groupby_fields[0]
                assert f'{identity_var}:1' in group_str

    def metadata_to_group(self, metadata, return_indices=False):
        if isinstance(self.groupers, list):
            all_groups = []
            for i in range(len(self.groupers)):
                grouper = self.groupers[i]
                groups = grouper.metadata_to_group(metadata) # n x 1
                all_groups.append(groups)
            all_groups = torch.stack(all_groups).T # n x len(groupers)

            new_groups = []
            for groups in all_groups: # for each instance
                new_group = []
                for i in range(len(self.groupers)):
                    grouper = self.groupers[i]
                    g = groups[i]
                    identity_var = grouper.groupby_fields[0]
                    group_str = grouper.group_field_str(g)
                    if f'y:0' in group_str:
                        label_idx = 0
                    else:
                        label_idx = 1
                    # detected the mention of an identity
                    if f'{identity_var}:1' in group_str:
                        g = i * 2 + label_idx
                        new_group.append(g)
                if len(new_group) == 0:
                    new_g = len(self.groupers) * 2 + label_idx
                else:
                    if self.sample_groups:
                        new_g = np.random.choice(new_group)
                    else:
                        new_g = new_group
                new_groups.append(new_g)
            if self.sample_groups:
                new_groups = torch.tensor(new_groups)
            return new_groups
        else:
            return self.groupers.metadata_to_group(metadata)


def get_grouper(dataset, multiple_groupers=False):
    try:
        grouper = dataset._eval_grouper
    except:
        if dataset._dataset_name == 'civilcomments':
            if multiple_groupers:
                grouper = dataset._eval_groupers
            else:
                grouper = CombinatorialGrouper(dataset, groupby_fields=['identity_any', 'y'])
        else:
            raise ValueError(f"Not sure how to address the grouper for dataset: {dataset._dataset_name}")
    return OneGrouper(grouper)


def get_classifier(model):
    if 'bert' in model.__class__.__name__.lower():
        classifier = deepcopy(model.classifier)
    else:
        classifier = deepcopy(model.fc)
    return classifier


def featurize_dataset(dataset, featurizer, args):
    featurizer.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.preprocess_bs, shuffle=False, num_workers=args.preprocess_workers)
    all_embeddings = []
    all_y = []
    all_metadata = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            x, y, metadata = batch[0].cuda(), batch[1], batch[2]
            embeddings = featurizer(x).detach().cpu()
            all_embeddings.append(embeddings)
            all_y.append(deepcopy(y)) # deepcopy is to fix a strange Runtime error of too many open files in dataloader 
            all_metadata.append(deepcopy(metadata)) # deepcopy is to fix a strange Runtime error of too many open files in dataloader 
    all_embeddings = torch.cat(all_embeddings)
    all_y = torch.cat(all_y)
    all_metadata = torch.cat(all_metadata)
    return all_embeddings, all_y, all_metadata


def get_eval_fn(eval_fn, y, metadata):
    return lambda y_pred: evaluate_w_metadata(eval_fn, y_pred, y, metadata)


def evaluate_w_metadata(eval_fn, y_pred, y, metadata):
    results = eval_fn(y_pred, y, metadata)
    wg_result = None
    for key, value in results.items():
        if key.endswith('_wg'):
            wg_result = value
            break
    return results, wg_result