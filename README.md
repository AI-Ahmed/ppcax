# PPCAx – Probabilistic PCA with JAX

## Overview

Probabilistic Principal Component Analysis (PPCA) model using DeepMind's JAX library. The model is a robust feature extraction and dimensionality reduction technique for high-dimensional, sparse multivariate data.

PPCA is a probabilistic approach to Principal Component Analysis (PCA), which allows for imputing missing values and estimating latent features in the data. By leveraging the power of JAX, this implementation ensures efficient and scalable computation, making it suitable for large-scale financial datasets.

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
    └── test_ppcax.py

25 directories, 17 files
```

## 🛠️ Installation and Setup Instructions

### Prerequisites

- **Python**: Ensure you have Python **3.10** or newer installed on your system.

### Installation Steps

1. **Clone the Repository**

   ```shell
   git clone https://github.com/AI-Ahmed/ppcax.git
   cd ppcax
   ```

2. **Install Flit**

   If you don't already have Flit installed, install it using `pip`:

   ```shell
   pip install flit
   ```

3. **Install the Package and Dependencies**

   Install the package along with its dependencies using Flit:

   ```shell
   flit install --deps develop
   ```

   This command installs the `ppcax` package along with all required dependencies, including development and testing tools like `pytest` and `flake8`.

### Alternative: Install Directly from GitHub

If you prefer to install the package directly from GitHub without cloning the repository:

```shell
pip install git+https://github.com/AI-Ahmed/ppcax
```

This command installs the latest version of `ppcax` from the main branch.

### Importing the Package

After installation, you can import the PPCA model in your Python code:

```python
from ppcax import PPCA
```

## 🧪 Running Tests

To run the unit tests and ensure everything is working correctly:

1. **Navigate to the Project Directory**

   If you haven't already, navigate to the project's root directory:

   ```shell
   cd ppcax
   ```

2. **Run Tests Using pytest**

   ```shell
   pytest tests/
   ```

   This command runs all tests located in the `tests/` directory.

## 📏 Code Style and Linting

To maintain code quality and consistency, `flake8` is used for linting. You can run linting checks with:

```shell
flake8 src/ tests/
```

Ensure that your code adheres to the style guidelines before committing changes.

## 📚 Usage Example

Here's a simple example of how to use the `PPCA` class:

```python
import numpy as np
from ppcax import PPCA

# Generate some sample data
data = np.random.rand(100, 1000)

# Create a PPCA model instance
ppca_model = PPCA(q=150)

# Fit the model to the data
ppca_model.fit(data, use_em=True)

# Transform the data to the lower-dimensional space
transformed_data = ppca_model.transform(lower_dim_only=True)

print("Transformed Data Shape:", transformed_data.shape)
```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE), which is a permissive open-source license that grants users extensive rights to use, modify, and distribute the software. See the [LICENSE](LICENSE) file for more details.

## 📣 Cite Our Work

If you find this work useful in your research, please consider citing:

```bibtex
@article{Atwa2024,
  author    = {Ahmed Atwa and Ahmed Sedky and Mohamed Kholief},
  title     = {Probabilistic PCA in High Dimensions: Stochastic Dimensionality Reduction on Sparse Multivariate Assets' Bars at High-Risk Regimes},
  journal   = {SSRN Electronic Journal},
  year      = {2024},
  note      = {Available at SSRN: \url{https://ssrn.com/abstract=4874874} or \url{http://dx.doi.org/10.2139/ssrn.4874874}}
}
```

---

## 🔧 Development Setup

If you're planning to contribute to the project or modify the code, follow these steps to set up your development environment:

1. **Clone the Repository**

   ```shell
   git clone https://github.com/AI-Ahmed/ppcax.git
   cd ppcax
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies:

   ```shell
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Flit**

   ```shell
   pip install flit
   ```

4. **Install the Package in Editable Mode**

   ```shell
   flit install --deps develop --symlink
   ```

   The `--symlink` option installs the package in editable mode, so changes to the code are immediately reflected without reinstallation.

5. **Install Pre-commit Hooks (Optional)**

   If you use `pre-commit` for code formatting and linting:

   ```shell
   pip install pre-commit
   pre-commit install
   ```

6. **Run Tests**

   ```shell
   pytest tests/
   ```

7. **Run Linting**

   ```shell
   flake8 src/ tests/
   ```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## 📬 Contact

For any questions or inquiries, please contact [Ahmed Nabil Atwa](mailto:your-email@example.com).

---

## 📝 Changelog

Refer to the [CHANGELOG](CHANGELOG.md) for details on updates and changes to the project.

---

## 📦 Publishing to PyPI (Maintainers Only)

To publish a new version of the package to PyPI:

1. **Update the Version Number**

   Increment the version number in `pyproject.toml`.

2. **Build the Package**

   ```shell
   flit build
   ```

3. **Publish to PyPI**

   ```shell
   flit publish
   ```

---

## 🌐 Links

- **Documentation**: [Link to documentation if available]
- **Issue Tracker**: [GitHub Issues](https://github.com/AI-Ahmed/ppcax/issues)
- **Source Code**: [GitHub Repository](https://github.com/AI-Ahmed/ppcax)