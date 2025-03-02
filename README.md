# Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions

This repository provides the source code for group-robust sample reweighting (GSR), a method for improving robustness to subpopulation shifts, developed in the paper [Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions](https://openreview.net/forum?id=aQj9Ifxrl6) by Qiao et al. in ICLR 2025. 

## Acknowledgments
We make use of the following great open-source repositories:
- [DFR](https://github.com/PolinaKirichenko/deep_feature_reweighting)
- [torch-influence](https://github.com/alstonlo/torch-influence)
- [WILDS](https://github.com/p-lambda/wilds)

Thank you!

## Setup
```sh
conda create -n gsr python=3.10   
conda activate gsr
pip install -r requirements.txt
```
Please manually install `torch-scatter==2.1.2` according to the torch and the cuda version of the device (e.g., `export CUDA=cu117`) with:
```sh
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
```

## Dataset Preparation

Run the following command to download and preprocess the datasets used by this repo. The default `root_dir` is `~/data`. 
```sh
python3 prepare_dataset.py --dataset <DATASET> --root_dir <DIR>
```

## Running the algorithm
We show the example for Waterbirds dataset. Please remember to specify `root_dir` if it differs from `~/data`.
### Stage 1: Representation learning. 

Select a held-out set from the training set, then train a base model on the remaining data.
```sh
python3 train_basemodel.py --output_dir=release/waterbirds --pretrained_model \
   --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
   --eval_freq=1 --dataset waterbirds \
   --augment_data --held_out_ratio 0.1
```

You may also directly use the dataset splits (including held-out sets) and checkpoints that we used and trained. To download them:
```sh
wget https://github.com/qiaoruiyt/GSR/releases/download/v0.0.1/release.tar.gz
tar -xzf release.tar.gz 
```
We recommend putting them in the root folder of this project, such that models are accessible via `release/${dataset}/final_checkpoint.pt`. 

### Stage 2: Last-layer retraining with sample reweighting. 
Make sure to align the `base_dir` with the `output_dir` from base model training. 
```sh
python gsr_retraining.py --base_dir release/waterbirds --dataset waterbirds --dro_metric acc --inf exact --inner_lr 0.0001 --max_outer_iter 100  --model resnet50 --multiplicative_updates --normalize_weights --optimizer lbfgs --outer_grad_clip 1.0 --outer_lr 1.0 --outer_lr_scheduler step --temperature 0.1 --weight_decay 0.1 --workers 0 --no_wandb --seed 1
```

Please add the `--multiple_groupers` flag for CivilComments dataset. 

## Citation
Please cite our work at
```bibtex
@inproceedings{qiao2025grouprobust,
   title={Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions},
   author={Rui Qiao and Zhaoxuan Wu and Jingtan Wang and Pang Wei Koh and Bryan Kian Hsiang Low},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
}
```