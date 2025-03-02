import numpy as np
import random 
import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from transformers import BertTokenizerFast, DistilBertTokenizerFast
from wilds import get_dataset as get_wilds_dataset
from datasets import MNLIDataset


def get_dataset(dataset, root_dir, download):
    if dataset.lower() == 'mnli':
        dataset = MNLIDataset(root_dir=root_dir, download=download)
    else:
        dataset = get_wilds_dataset(dataset, root_dir=root_dir, download=download)
    return dataset


class MySubset(torch.utils.data.Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)


def set_numpy_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def split_dataset(dataset, ratio=0.1, seed=1):
    set_numpy_seed(seed)

    held_out_idx = np.random.choice(len(dataset), int(ratio * len(dataset)), replace=False)
    held_out_idx = np.sort(held_out_idx)

    train_idx = np.setdiff1d(np.arange(len(dataset)), held_out_idx)
    return held_out_idx, train_idx
    

def get_transform(dataset, args):
    if dataset.lower() in ['waterbirds', 'celeba']:
        target_resolution = (224, 224)
        train_transform = get_transform_cub(target_resolution=target_resolution,
                                            train=True, augment_data=args.augment_data)
        test_transform = get_transform_cub(target_resolution=target_resolution,
                                        train=False, augment_data=args.augment_data)
    elif dataset.lower() in ['civilcomments']:
        train_transform = initialize_bert_transform(args)
        test_transform = initialize_bert_transform(args)
    elif dataset.lower() in ['mnli']: # already pretokenized with bert
        train_transform, test_transform = None, None
    else:
        raise ValueError(f'Dataset {dataset} not recognized.')
    return train_transform, test_transform


def get_transform_cub(target_resolution, train, augment_data):
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loader(data, train, reweight_groups, reweight_classes, reweight_places, **kwargs):
    if not train: # Validation or testing
        assert reweight_groups is None
        assert reweight_classes is None
        assert reweight_places is None
        shuffle = False
        sampler = None
    elif not (reweight_groups or reweight_classes or reweight_places): # Training but not reweighting
        shuffle = True
        sampler = None
    elif reweight_groups:
        # Training and reweighting groups
        # reweighting changes the loss function from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight)
        group_weights = len(data) / data.group_counts
        weights = group_weights[data.group_array]

        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    elif reweight_classes:  # Training and reweighting classes
        class_weights = len(data) / data.y_counts
        weights = class_weights[data.y_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    else: # Training and reweighting places
        place_weights = len(data) / data.p_counts
        weights = place_weights[data.p_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False

    loader = DataLoader(
        data,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)
    return loader


def log_data(logger, train_data, test_data, val_data=None, get_yp_func=None):
    logger.write(f'Training Data (total {len(train_data)})\n')
    # group_id = y_id * n_places + place_id
    # y_id = group_id // n_places
    # place_id = group_id % n_places
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n')
    logger.write(f'Test Data (total {len(test_data)})\n')
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n')
    if val_data is not None:
        logger.write(f'Validation Data (total {len(val_data)})\n')
        for group_idx in range(val_data.n_groups):
            y_idx, p_idx = get_yp_func(group_idx)
            logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n')



def initialize_bert_transform(config):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in config.model
    assert config.max_token_length is not None

    tokenizer = get_bert_tokenizer(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_token_length,
            return_tensors="pt",
        )
        if config.model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif config.model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform