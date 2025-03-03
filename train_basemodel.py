"""The code is modified from the DFR repo."""

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
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p, get_training_args, get_model, get_optimizer, get_grouper
from utils import update_dict, get_results, write_dict_to_tb

from data_utils import get_transform, get_dataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import get_train_loader, get_eval_loader

args = get_training_args()
print(args)
set_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
logger = Logger(os.path.join(args.output_dir, 'log.txt'))
writer = SummaryWriter(log_dir=args.output_dir)

train_transform, test_transform = get_transform(args.dataset, args)
dataset = get_dataset(dataset=args.dataset, root_dir=args.root_dir, download=args.download)
grouper = get_grouper(dataset)

train_dataset = dataset.get_subset("train", transform=train_transform)
val_dataset = dataset.get_subset("val", transform=test_transform)
test_dataset = dataset.get_subset("test", transform=test_transform)
testset_dict = {'test': test_dataset, 
        'val': val_dataset}


if args.held_out_ratio > 0:
    from data_utils import MySubset, split_dataset
    held_out_idx, train_idx = split_dataset(train_dataset, args.held_out_ratio)
    train_dataset = MySubset(train_dataset, train_idx)
    held_out_set = MySubset(train_dataset, held_out_idx)

    np.save(os.path.join(args.output_dir, "held_out_idx.npy"), held_out_idx)

if args.limit is not None:
    args.batch_size = min(args.batch_size, args.limit)

# Construct data loader
loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
train_loader = get_train_loader('standard', train_dataset, **loader_kwargs)
test_loader_dict = {name: get_eval_loader('standard', test_dataset, **loader_kwargs) for name, test_dataset in testset_dict.items()}

# Load model
model = get_model(args.model, n_classes=train_dataset.n_classes)
model.cuda()

if args.optimizer != 'adam' and 'bert' in args.model:
    print("Warning: not using Adam optimizer for BERT model")
optimizer = get_optimizer(model.parameters(), args)
if args.scheduler:
    print("Using scheduler")
    if 'bert' in args.model:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=args.num_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
else:
    scheduler = None

criterion = torch.nn.CrossEntropyLoss()

for epoch in range(args.num_epochs):
    model.train()
    loss_meter = AverageMeter()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(grouper.n_groups)}

    all_y_pred = []
    all_y_true = []
    all_metadata = []
    for batch in tqdm.tqdm(train_loader):
        x, y, metadata = batch
        g = grouper.metadata_to_group(metadata)
        x, y = x.cuda(), y.cuda() 

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss, x.size(0))
        update_dict(acc_groups, y, g, logits)

        y_pred = logits.argmax(dim=1)
        all_y_pred.append(y_pred.cpu())
        all_y_true.append(y.cpu())
        all_metadata.append(metadata)

    all_y_pred = torch.cat(all_y_pred)
    all_y_true = torch.cat(all_y_true)
    all_metadata = torch.cat(all_metadata)
    results, _ = dataset.eval(all_y_pred, all_y_true, all_metadata)
    write_dict_to_tb(writer, results, "train/", epoch)
    logger.write("Train results \n")
    logger.write(str(results))

    if args.scheduler:
        scheduler.step()
    logger.write(f"Epoch {epoch}\t Loss: {loss_meter.avg}\n")

    if epoch % args.eval_freq == 0:
        # Iterating over datasets we test on
        for test_name, test_loader in test_loader_dict.items():
            model.eval()
            all_y_pred = []
            all_y_true = []
            all_metadata = []
            for batch in tqdm.tqdm(test_loader):
                x, y, metadata = batch
                g = grouper.metadata_to_group(metadata)
                x, y = x.cuda(), y.cuda()

                with torch.no_grad():
                    logits = model(x)
                    y_pred = logits.argmax(dim=1)
                    all_y_pred.append(y_pred.cpu())
                    all_y_true.append(y.cpu())
                    all_metadata.append(metadata)

            all_y_pred = torch.cat(all_y_pred)
            all_y_true = torch.cat(all_y_true)
            all_metadata = torch.cat(all_metadata)
            results, _ = dataset.eval(all_y_pred, all_y_true, all_metadata)
            tag = test_name
            write_dict_to_tb(writer, results, "test_{}/".format(tag), epoch)
            logger.write("Test results \n")
            logger.write(str(results))

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'tmp_checkpoint.pt'))
        
    logger.write('\n')

torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_checkpoint.pt'))
