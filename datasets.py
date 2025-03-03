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