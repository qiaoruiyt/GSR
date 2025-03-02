import os, sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from wilds.common.data_loaders import get_train_loader, get_eval_loader 
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper

class MNLIDataset(WILDSDataset):
    """
    MultiNLI dataset.
    """

    _dataset_name = 'mnli'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz',
            'compressed_size': 40_604_486 }}
    

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='unofficial'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        sys.path.append(self.data_dir)
        if download:
            import wget
            version_file = os.path.join(self.data_dir, f'RELEASE_v{self.version}.txt')
            # write version file
            with open(version_file, 'w') as f:
                f.write('placeholder')
            if not os.path.exists(os.path.join(self.data_dir, 'metadata_random.csv')):
                wget.download('https://raw.githubusercontent.com/kohpangwei/group_DRO/master/dataset_metadata/multinli/metadata_random.csv', self.data_dir)
            if not os.path.exists(os.path.join(self.data_dir, 'utils_glue.py')):
                wget.download('https://raw.githubusercontent.com/kohpangwei/group_DRO/master/utils_glue.py', self.data_dir)
        self.root_dir = root_dir

        confounder_names = ['sentence2_has_negation']

        # Read in metadata
        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, f"metadata_random.csv"))

        # Get the y values
        self._y_array = torch.LongTensor(self.metadata_df['gold_label'].values)
        self._y_size = 1
        self._n_classes = 3

        # Get metadata
        self.confounder = self.metadata_df['sentence2_has_negation'].values

        self._metadata_array = torch.stack(
            [torch.LongTensor(self.confounder), self._y_array],
            dim=1)
        self._metadata_fields = confounder_names + ['y']
        self._metadata_map = {
            'y': ['contradiction', '   entailment', '      neutral'] # Padding for str formatting
        }

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(confounder_names + ['y']))

        self._split_scheme = split_scheme
        # if self._split_scheme != 'official':
        #     raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = self.metadata_df["split"]

        # Extract features
        self.features_array = []
        for feature_file in [
            'cached_train_bert-base-uncased_128_mnli',  
            'cached_dev_bert-base-uncased_128_mnli',
            'cached_dev_bert-base-uncased_128_mnli-mm'
            ]:

            features = torch.load(
                os.path.join(
                    self.data_dir,
                    feature_file))

            self.features_array += features

        self.all_input_ids = torch.tensor([f.input_ids for f in self.features_array], dtype=torch.long)
        self.all_input_masks = torch.tensor([f.input_mask for f in self.features_array], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features_array], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_id for f in self.features_array], dtype=torch.long)

        self.x_array = torch.stack((
            self.all_input_ids,
            self.all_input_masks,
            self.all_segment_ids), dim=2)

        assert torch.all(self.all_label_ids == self.y_array)

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns the input text for a given index.
        """
        return self.x_array[idx]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels 
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)


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
