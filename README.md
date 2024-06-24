# PPCAx – Probabilistic PCA with JAX

## Overview
Probabilistic Principal Component Analysis (PPCA) model using DeepMind's JAX library. The model is a robust feature extraction and dimensionality reduction technique for high-dimensional, sparse multivariate data.

PPCA is a probabilistic approach to Principal Component Analysis (PCA), which allows for imputing missing values and estimating latent variables in the data. By leveraging the power of JAX, this implementation ensures efficient and scalable computation, making it suitable for large-scale financial datasets.

The methodology used in this project was initially proposed in our research manuscript titled *"Probabilistic PCA in High Dimensions: Stochastic Dimensionality Reduction on Sparse Multivariate Assets' Bars at High-Risk Regimes"*. This work presents a novel approach for analyzing portfolio behavior during periods of high market turbulence and risk by:

1. Using information-driven bar techniques to synchronize and sample imbalanced sequence volumes.
2. Applying a sampling event-based technique, the CUMSUM Filtering method, to create strategic trading plans based on volatility.
3. Employing an improved version of the Gaussian Linear System called PPCA for feature extraction from the latent space.

Our findings suggest that PPCA is highly effective in estimating sparse data and forecasting the effects of individual assets within a portfolio under varying market conditions. This repository contains the core implementation of the PPCA model, demonstrating its capability to establish significant relationships among correlated assets during high-risk regimes.
## 📁 Directory Structure

```bash
.
├── LICENSE
├── README.md
├── config
├── data
│   ├── bars
│   ├── metadata
│   ├── sample
│   │   ├── r1
│   │   └── r2
│   └── tickers
├── models
├── notebooks
├── pyproject.toml
├── reports
│   ├── docs
│   ├── eval
│   ├── figures
│   └── train
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── eval
│   ├── ft_eng
│   ├── ppcax
│   │   ├── __init__.py
│   │   └── _ppcax.py
│   ├── preprocessing
│   └── utils
└── tests
    ├── __init__.py
    ├── gen_data.py
    └── test.py

25 directories, 17 files
```

## 🛠️ Initialization Instructions

Begin with the project setup process:
- Ensure Python version 3.10 or newer is installed

1. Prepare your Python environment and install all required packages with the `pyproject.toml` file. Alternatively, for a manual approach, execute the following to install dependencies:
```shell
git clone https://github.com/AI-Ahmed/ppcax.git
cd ppcax
pip install -r requirements.txt
```

2. Execute unit tests by running:
```shell
pytest tests/test.py
```

3. If you want to install this project as a package, use the command below:
```shell
pip install git+https://github.com/AI-Ahmed/ppcax
```
Then, in Python, you can import the model,
```python
from ppcax import PPCA
```

## Cite Our Work

If you find this work useful in your research, please consider citing:

```bibtex
@article{Atwa2023,
  author    = {Ahmed Atwa and Mohamed Kholief and Ahmed Sedky},
  title     = {Probabilistic PCA in High Dimensions: Stochastic Dimensionality Reduction on Sparse Multivariate Assets' Bars at High-Risk Regimes},
  journal   = {SSRN Electronic Journal},
  year      = {2024},
  note      = {Available at SSRN: \url{https://ssrn.com/abstract=4874874} or \url{http://dx.doi.org/10.2139/ssrn.4874874}}
}

```

## 📄 License
This project is licensed under the [Apache License 2.0](LICENSE), which is a permissive open-source license that grants users extensive rights to use, modify, and distribute the software. See the [LICENSE](LICENSE) file for more details.
