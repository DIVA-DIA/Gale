#!/bin/bash

conda create -y -n gale python=3.7.4 colorlog numpy pandas tqdm matplotlib seaborn pytorch torchvision cudatoolkit=10.1 psutil -c pytorch conda-forge
source activate gale
pip install sigopt tensorboardx darwin-py wandb

# Extra stuff for maphd branch
pip install git+https://github.com/NarayanSchuetz/SpectralLayersPyTorch
conda install scikit-learn
