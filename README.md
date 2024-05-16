# ppcax – Probabilistic PCA with JAX

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
├── reports
│   ├── docs
│   ├── eval
│   ├── figures
│   └── train
├── requirements.txt
└── src
    ├── __init__.py
    ├── eval
    ├── ft_eng
    ├── modeling
    │   ├── __init__.py
    │   └── ppcax.py
    ├── preprocessing
    └── utils

21 directories, 6 files
```

## 🛠️ Getting Started

To get started with this project:

1. Clone this repository to your local machine.
```bash
pip install 
```
2. Set up your Python environment and install the necessary dependencies listed in `requirements.txt`.
3. Explore the provided directories to understand the structure of the project.

## 💡 Usage

This project template provides a flexible framework for organizing and managing financial machine learning projects. Here are some ways you can use it:

- **Data Management**: Store raw and processed financial data in the `data/` directory.
- **Modeling**: Develop and train machine learning models in the `src/` directory.
- **Notebooks**: Use Jupyter notebooks in the `notebooks/` directory for exploratory data analysis and experimentation.
- **Reporting**: Generate reports and visualizations in the `reports/` directory to communicate your findings.

## 🤝 Contributing

Contributions to this project template are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## 📄 License
This project is licensed under the [Apache License 2.0](LICENSE), which is a permissive open-source license that grants users extensive rights to use, modify, and distribute the software. See the [LICENSE](LICENSE) file for more details.
