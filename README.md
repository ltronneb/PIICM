[![DOI](https://zenodo.org/badge/465453049.svg)](https://zenodo.org/badge/latestdoi/465453049)

# PIICM
Permutation Invariant multi-output Gaussian Process regression for dose-response prediction


## Installation
The steps below sets up a conda environment and installs the required version of  PyTorch. It then clones the repo, and installs PIICM.

```{python}
conda create --name piicm python=3.9
conda activate piicm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
git clone https://github.com/ltronneb/PIICM.git
cd PIICM/gpytorch
pip install .
```

To replicate the results from the paper, the scripts for pre-processing the dataset are found in preprocessing/ and the script for formatting the data and running models are in model/
