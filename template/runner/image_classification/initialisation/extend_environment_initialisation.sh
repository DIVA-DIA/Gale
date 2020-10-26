#!/bin/bash

source activate gale

echo "Installation started..."
# install packages
pip install spectralLayersPyTorch
conda install scikit-learn -y