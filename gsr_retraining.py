"""Evaluate DFR on spurious correlations datasets."""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from functools import partial
import pickle
import time

from utils import Logger, AverageMeter, set_seed, evaluate, add_label_noise
from data_utils import get_dataset, get_transform, MySubset, split_dataset
from utils import get_last_layer_retraining_args, get_model, get_classifier, get_feature_extractor, featurize_dataset, get_grouper, get_predictions
from gsr import GSRTrainer, SubpopDataset, IndexedDataset
from baselines import Trainer

args = get_last_layer_retraining_args()

train_transform, test_transform = get_transform(args.dataset, args)
dataset = get_dataset(dataset=args.dataset, root_dir=args.root_dir, download=False)

train_dataset = dataset.get_subset("train", transform=train_transform)
all_val_dataset = dataset.get_subset("val", transform=test_transform)
test_dataset = dataset.get_subset("test", transform=test_transform)

heldout_path = os.path.join(args.base_dir, "held_out_idx.npy")
ckpt_path = os.path.join(args.base_dir, "final_checkpoint.pt")
heldout_idx = np.load(heldout_path)
if args.no_heldout:
    all_idx = np.arange(len(train_dataset))
    train_idx = np.setdiff1d(all_idx, heldout_idx)
    train_dataset = MySubset(train_dataset, train_idx)
    print(f"Retraining on the training set of size {len(train_idx)}")
else:
    train_dataset = MySubset(train_dataset, heldout_idx)
    print(f"Retraining on the heldout training set of size {len(heldout_idx)}")

# Load model
n_classes = train_dataset.n_classes
model = get_model(args.model, n_classes)
model.load_state_dict(torch.load(ckpt_path))
model.cuda()
model.eval()

classifier = get_classifier(model)
featurizer = get_feature_extractor(model)

train_cache_file = os.path.join(args.base_dir, f'train_cache_{args.model}.pkl')
val_cache_file = os.path.join(args.base_dir, f'val_cache_{args.model}.pkl')
test_cache_file = os.path.join(args.base_dir, f'test_cache_{args.model}.pkl')

def load_or_save_cache(cache_file, dataset, featurizer, args):
    if not os.path.exists(cache_file):
        x, y, metadata = featurize_dataset(dataset, featurizer, args)
        with open(cache_file, 'wb') as f:
            pickle.dump((x, y, metadata), f)
    else:
        print(f"Loading cache from {cache_file}")
        with open(cache_file, 'rb') as f:
            x, y, metadata = pickle.load(f)
    return x, y, metadata

print("Featurizing datasets ...")
if args.no_cache:
    train_x, train_y, train_metadata = featurize_dataset(train_dataset, featurizer, args)
    all_val_x, all_val_y, all_val_metadata = featurize_dataset(all_val_dataset, featurizer, args)
    test_x, test_y, test_metadata = featurize_dataset(test_dataset, featurizer, args)
else:
    train_x, train_y, train_metadata = load_or_save_cache(train_cache_file, train_dataset, featurizer, args)
    all_val_x, all_val_y, all_val_metadata = load_or_save_cache(val_cache_file, all_val_dataset, featurizer, args)
    test_x, test_y, test_metadata = load_or_save_cache(test_cache_file, test_dataset, featurizer, args)
print(f"Train set shape: {train_x.shape}")


if args.val_subsample is not None:
    from data_utils import split_dataset
    if args.val_subsample == 0:
        raise ValueError("val_subsample cannot be 0")
    elif args.val_subsample == 1:
        val_idx1 = np.arange(len(all_val_dataset))
    else:
        val_idx1, val_idx2 = split_dataset(all_val_dataset, args.val_subsample) # seed is automatically applied for reproducibility
    print(f"Using a subsample of val set of size {len(val_idx1)}")
else:
    val_idx1 = np.arange(len(all_val_dataset))
    val_idx2 = None
set_seed(args.seed)

grouper = get_grouper(dataset, args.multiple_groupers)
train_g = grouper.metadata_to_group(train_metadata)
all_val_g = grouper.metadata_to_group(all_val_metadata)
test_g = grouper.metadata_to_group(test_metadata)

if args.class_balanced:
    cb_groups = all_val_y
    all_val_g = cb_groups
if args.mis_balanced:
    # get groups based on misclassification
    all_val_dataset = SubpopDataset(all_val_x, all_val_y, all_val_g, all_val_metadata)
    y_preds, y_true = get_predictions(classifier, all_val_dataset)
    y_correct = y_preds == y_true
    mb_groups = y_correct.long()
    print("# of correct predictions:", mb_groups.sum().item())
    print("# of incorrect predictions:", len(y_correct) - mb_groups.sum().item())
    all_val_g = mb_groups
if args.class_balanced and args.mis_balanced:
    # create new groups based on both class and misclassification
    all_val_g = cb_groups * 2 + mb_groups
    group_counts = torch.bincount(all_val_g)
    print("Group counts:", group_counts)

if args.val_fraction is not None:
    set_seed(args.seed)
    val_idx1 = np.random.choice(val_idx1, int(args.val_fraction * len(val_idx1)), replace=False)
    val_idx2 = np.random.choice(val_idx2, int(args.val_fraction * len(val_idx2)), replace=False)
    print(f"Using a fraction of val set of size {len(val_idx1)} out of {len(all_val_dataset)}")
    print(f"Using a fraction of target set of size {len(val_idx2)} out of {len(all_val_dataset)}")
val_x, val_y, val_metadata, val_g = all_val_x[val_idx1], all_val_y[val_idx1], all_val_metadata[val_idx1], all_val_g[val_idx1]
if args.keep_val_group:
    val_g = grouper.metadata_to_group(val_metadata)

if args.train_val:
    # Use the balanced val set as the train set
    train_x, train_y, train_g = val_x, val_y, val_g
if args.gb_subsample:
    from baselines import subsample_group_balanced_dataset
    print("Subsampling group balanced dataset")
    train_x, train_y, train_g = subsample_group_balanced_dataset(train_x, train_y, train_g)
    print(f"Subsampled train set of size {len(train_x)}")
if args.label_noise is not None and args.label_noise > 0:
    print(f"Adding label noise with noise rate {args.label_noise}")
    new_train_y, noise_indices = add_label_noise(train_y, args.label_noise)
    actual_noise_rate = (new_train_y != train_y).sum().item() / len(train_y)
    print(f"Actual label noise rate: {actual_noise_rate}")
    train_y = new_train_y
if args.group_noise is not None and args.group_noise > 0:
    print(f"Adding group noise with noise rate {args.group_noise}")
    new_train_g, noise_indices = add_label_noise(train_g, args.group_noise)
    actual_noise_rate = (new_train_g != train_g).sum().item() / len(train_g)
    print(f"Actual group noise rate: {actual_noise_rate}")
    train_g = new_train_g

train_group_indices = None
val_group_indices = None
test_group_indices = None
if args.multiple_groupers:
    print("Using multiple groupers and procesing group indices")
    # train_group_indices = grouper.metadata_to_group_indices(train_metadata)
    val_group_indices = grouper.metadata_to_group_indices(val_metadata)
    test_group_indices = grouper.metadata_to_group_indices(test_metadata)

train_dataset = SubpopDataset(train_x, train_y, train_g, train_metadata, all_group_indices=train_group_indices)
val_dataset = SubpopDataset(val_x, val_y, val_g, val_metadata, all_group_indices=val_group_indices)
test_dataset = SubpopDataset(test_x, test_y, test_g, test_metadata, all_group_indices=test_group_indices)

# The dataset needs to be wrapped to include indices in each batch
train_dataset = IndexedDataset(train_dataset)
val_dataset = IndexedDataset(val_dataset)
test_dataset = IndexedDataset(test_dataset)

if args.eval_base:
    from utils import evaluate_group_metrics
    n_groups = train_dataset.n_groups
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    train_metrics = evaluate_group_metrics(classifier, train_loader)
    val_metrics = evaluate_group_metrics(classifier, val_loader)
    test_metrics = evaluate_group_metrics(classifier, test_loader)
    print("Train metrics:", train_metrics[-1])
    print("Val metrics:", val_metrics[-1])
    print("Test metrics:", test_metrics[-1])

if args.target_fraction is not None and args.val_fraction is None:
    set_seed(args.seed)
    val_idx2_frac = np.random.choice(val_idx2, int(args.target_fraction * len(val_idx2)), replace=False)
    print(f"Using a fraction of val set of size {len(val_idx2_frac)} out of {len(val_idx2)}")
    val_idx2 = val_idx2_frac

if not args.no_target and args.val_subsample is not None and args.val_subsample != 1:
    print(f"Retraining on a separate target set of size {len(val_idx2)}")
    target_x, target_y, target_metadata = all_val_x[val_idx2], all_val_y[val_idx2], all_val_metadata[val_idx2]
    target_g = all_val_g[val_idx2]
    target_group_indices = grouper.metadata_to_group_indices(target_metadata) if args.multiple_groupers else None
    target_dataset = SubpopDataset(target_x, target_y, target_g, target_metadata, all_group_indices=target_group_indices)
    target_dataset = IndexedDataset(target_dataset)
else:
    # use the val set as the target set, but this can result in overfitting to the val set
    target_x, target_y, target_metadata = val_x, val_y, val_metadata
    target_dataset = val_dataset
    print(f"Using val set of size {len(val_dataset)} as the target set")

if args.target_noise is not None and args.target_noise > 0:
    print(f"Adding target noise with noise rate {args.target_noise}")
    new_target_y, noise_indices = add_label_noise(target_y, args.target_noise)
    actual_noise_rate = (new_target_y != target_y).sum().item() / len(target_y)
    print(f"Actual target noise rate: {actual_noise_rate}")
    target_y = new_target_y
    target_dataset = SubpopDataset(target_x, target_y, target_g, target_metadata, all_group_indices=target_group_indices)
    target_dataset = IndexedDataset(target_dataset)

writer = None
if not args.no_wandb:
    import wandb
    wandb_dir = os.path.expanduser("~/wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    writer = wandb.init(project='gsr', group=args.runs_name, config=args, dir=wandb_dir)

set_seed(args.seed)
model = torch.nn.Linear(train_x.shape[1], n_classes)

train_results = None
if args.group_dro:
    trainer = Trainer(args, model, train_dataset, target_dataset=target_dataset, val_dataset=val_dataset, test_dataset=test_dataset, writer=writer)
    raise NotImplementedError
elif args.group_balanced:
    trainer = Trainer(args, model, train_dataset, target_dataset=target_dataset, val_dataset=val_dataset, test_dataset=test_dataset, writer=writer)
    sample_weights = trainer.get_group_balanced_weights()
    train_groups_weights = trainer.get_dataset_group_weights(train_dataset, sample_weights)
    print("Train group weights:", train_groups_weights)
    train_results, best_model = trainer.train(sample_weights, resample=args.resample)
    print("Train info:", train_results)
    best_val = train_results['best_val'] # used for model selection

    y_preds, y_true = get_predictions(best_model, val_dataset)
    official_results, official_results_str = dataset.eval(y_preds, y_true, val_metadata) 
    print("Official results:", official_results_str)
else:
    trainer = GSRTrainer(args, model, train_dataset, target_dataset=target_dataset, val_dataset=val_dataset, test_dataset=test_dataset, writer=writer)
    if args.verbose:
        start_time = time.time()
    train_results, best_w, best_model = trainer.solve()
    if args.verbose:
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds with iterations {args.max_outer_iter}")
    best_train, best_val, best_test = train_results['best_train_acc'], train_results['best_val_wg'], train_results['best_test_wg']

    print(f"Best train: {best_train}, Best val: {best_val}, Best test: {best_test}")

y_preds, y_true = get_predictions(best_model, test_dataset)
official_results, official_results_str = dataset.eval(y_preds, y_true, test_metadata) 

print("Test results:", official_results_str)
if 'adj_acc_avg' in official_results.keys(): # For waterbirds because the val/test sets are balanced
    avg_test_acc = official_results['adj_acc_avg']
else:
    avg_test_acc = official_results['acc_avg']
wg_test_acc = official_results['acc_wg']
hparams = vars(args)
result_json = {
    'val_acc': best_val,
    'avg_test_acc': avg_test_acc,
    'wg_test_acc': wg_test_acc, 
    'seed': args.seed,
    'hparams': hparams,
    'all_results': train_results,
}
if train_results is not None:
    result_json['avg_test_acc_unofficial'] = train_results['best_test_acc']
    result_json['wg_test_acc_unofficial'] = train_results['best_test_wg']

os.makedirs(args.output_dir, exist_ok=True)
mode = 'a+'
with open(os.path.join(args.output_dir, 'results.json'), mode) as f:
    f.write(json.dumps(result_json) + '\n')

with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    f.write('done')

# if args.save_stats:
group_file = os.path.join(args.output_dir, 'group_stats.pkl')
weights_file = os.path.join(args.output_dir, 'weights.pkl')
with open(weights_file, 'wb') as f:
    pickle.dump(best_w.cpu().numpy(), f)
with open(group_file, 'wb') as f:
    pickle.dump(train_g.cpu().numpy(), f)

if writer is not None:
    writer.log({"group_stats": train_g})
    writer.log({"weights": best_w})

try:
    # save noise indices
    noise_file = os.path.join(args.output_dir, 'noise_indices.pkl')
    with open(noise_file, 'wb') as f:
        pickle.dump(noise_indices, f)
except NameError:
    pass



