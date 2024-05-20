# PPCAx – Probabilistic PCA with JAX

## Overview

In the current exploration of financial data analysis, we introduce two distinct versions of Probabilistic Principal Component Analysis (PPCA): a traditional PPCA framework and an enhanced "PPCA in High Dimension" model tailored for the vast dimensionality of information-driven datasets such as Dollar Bars Runs (DBRs). Traditional PPCA offers a stochastic approach for dimensionality reduction and feature extraction, which is critical for understanding high-risk market dynamics. However, its standard application presents limitations when faced with the intricacies of high-dimensional financial data. It is where our "PPCA in High Dimension" model gains prominence. It is meticulously crafted not merely to cope with but also to harness and elucidate the complexity inherent in such expansive datasets, which conventional PPCA cannot adequately manage. Both models are further refined through the integration with the JAX eco-systems, which provides the computational finesse needed to process large-scale data with heightened speed and precision via GPU acceleration.

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
@article{Atwa2024PPCA,
  title={Probabilistic PCA: Stochastic Dimensionality Reduction of Sparse Multivariate Assets’ Bars at High-Risk Regimes},
  author={Atwa, Ahmed N. and Kholief, Mohamed and Sedky, Ahmed},
  journal={Journal of Financial Econometrics},
  volume={XX},
  number={X},
  pages={1-31},
  year={2024},
  month={Month},
  publisher={Oxford University Press},
  doi={10.1093/XXXXXX/XXXXXX},
  url={https://doi.org/10.1093/XXXXXX/XXXXXX}
}
```

## 📄 License
This project is licensed under the [Apache License 2.0](LICENSE), which is a permissive open-source license that grants users extensive rights to use, modify, and distribute the software. See the [LICENSE](LICENSE) file for more details.
