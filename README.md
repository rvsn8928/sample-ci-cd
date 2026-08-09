# Employee Churn Prediction — Python Repository

This repository converts `Learner_Notebook3.ipynb` into a modular Python project for **Predict Employee Churn with Decision Trees and Random Forests**.

## What is included

- Data loading and preprocessing
- One-hot encoding for `department` and `salary`
- Stratified train/test split
- Decision Tree classifier
- Random Forest classifier
- Training/test accuracy and classification metrics
- ROC-AUC when probability predictions support it
- Turnover-frequency EDA charts
- Decision-tree Graphviz exports
- Unit tests

The original notebook's interactive `ipywidgets` controls are represented as command-line parameters in `scripts/train.py` so the workflow can run without Jupyter.

## Project structure

```text
employee_churn_repo/
├── data/
│   └── employee_data.csv        # add the dataset here
├── assets/images/               # optional notebook assets
├── outputs/                     # generated charts/models/metrics
├── scripts/
│   ├── eda.py
│   └── train.py
├── src/employee_churn/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── visualization.py
├── tests/
│   └── test_data.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

For tree PNG generation, install the **Graphviz system package** as well as the Python package. For Ubuntu/Debian:

```bash
sudo apt-get install graphviz
```

## Add the dataset

Place the original notebook dataset at:

```text
data/employee_data.csv
```

The code expects a target column named `quit` and, for the notebook's categorical encoding, columns named `department` and `salary`.

## Run EDA

```bash
PYTHONPATH=src python scripts/eda.py
```

## Train both models

```bash
PYTHONPATH=src python scripts/train.py --depth 5 --trees 100
```

Generated artifacts are written to `outputs/`, including turnover plots, `metrics.json`, and tree visualizations when Graphviz is available.

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Notebook-to-repo mapping

| Notebook section | Python implementation |
|---|---|
| Import libraries | `requirements.txt`, module imports |
| Exploratory Data Analysis | `scripts/eda.py`, `visualization.py` |
| Encode categorical features | `data.py::encode_features` |
| Class imbalance | `scripts/eda.py` target distribution |
| Train/test split | `data.py::split_data` |
| Decision Tree | `models.py::train_decision_tree` |
| Random Forest | `models.py::train_random_forest` |
| Feature importance / metrics | `models.py::evaluate_model` and model attributes |
| Tree visualization | `visualization.py::save_tree` |
