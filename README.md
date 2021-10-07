# Senior Thesis
This is the main repo for my senior thesis. The goal of the project is to determine how to optimally learn a graph (network) structure on time series using graph dynamics and benchmark it against classical methods. This repo will contain all of the code files used for the project as an importable Python package

# Prerequisites

The necessary software needed to build the project will be listed here and will be expanded if necessary. As of right now, it is just

- [git](https://git-scm.com/)
- Late model [Anaconda](https://www.anaconda.com/products/individual)

## Conda Environment

It is advisable to install this project in a clean conda environment. This project is installed as an editable module, so anything in the `thesis` folder can be imported just as a regular Pyhton package. Dependencies and package information are listed in `setup.py`. In order to build the environment, run the following commands in the root directory of this repository:

```bash
conda create -n thesis python=3.8

conda activate thesis

pip install -e .
```

Once this is done, you can deactivate the environment with `conda deactivate` and reactivate it with `conda activate thesis`.
