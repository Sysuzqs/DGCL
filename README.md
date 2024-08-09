# DGCL: Dual-Graph Neural Networks Contrastive Learning for Molecular Property Prediction

This repository is the official implementation of **DGCL**.

## Environments

### Install via Conda

```bash
# Clone the environment
conda env create -f env.yml
# Activate the environment
conda activate dgcl
```

## Model

We have two implementations: DGCL and AttentionDGCL.

```bash
# For DGCL
cd DGCL
# For AttentionDGCL
cd Attention
```

## Pre-training

please run the following command:
```bash
python main_pretrain.py
```

## Fine-tuning

If you want to use our pre-trained model directly for molecular property prediction, please run the following command:

```bash
# For classification tasks
python main_gnn_classificaiton.py --task bbbp --random_seed 0
# For regression tasks
python main_gnn_regression.py --task lipo --random_seed 0
```