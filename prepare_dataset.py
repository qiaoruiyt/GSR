from wilds.datasets.celebA_dataset import CelebADataset
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from datasets import MNLIDataset
import argparse
import wget
import os

def download_and_prepare_dataset(dataset_name, root_dir, download=True):
    if dataset_name == 'celebA':
        dataset = CelebADataset(root_dir=root_dir, download=download)
        metadata_dir = dataset.data_dir
        wget.download('https://raw.githubusercontent.com/PolinaKirichenko/deep_feature_reweighting/main/celeba_metadata.csv', metadata_dir)
        os.rename(os.path.join(metadata_dir, 'celeba_metadata.csv'), os.path.join(metadata_dir, 'metadata.csv'))
    elif dataset_name == 'waterbirds':
        dataset = WaterbirdsDataset(root_dir=root_dir, download=download)
    elif dataset_name == 'civilcomments':
        dataset = CivilCommentsDataset(root_dir=root_dir, download=download)
    elif dataset_name == 'mnli':
        dataset = MNLIDataset(root_dir=root_dir, download=download)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--root_dir', type=str, default='~/data')
    args = parser.parse_args()
    args.root_dir = os.path.expanduser(args.root_dir)

    download_and_prepare_dataset(args.dataset, args.root_dir)