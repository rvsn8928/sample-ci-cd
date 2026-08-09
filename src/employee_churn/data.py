from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_DATA_PATH = Path("data/employee_data.csv")


def load_data(path=DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the employee churn dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add employee_data.csv under data/."
        )
    return pd.read_csv(path)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode department and salary, matching the notebook workflow."""
    result = df.copy()
    categorical = [c for c in ("department", "salary") if c in result.columns]
    for col in categorical:
        dummies = pd.get_dummies(result[col], prefix=col, dtype=int)
        result = result.join(dummies)
    if categorical:
        result = result.drop(columns=categorical)
    return result


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 0) -> Tuple:
    """Create stratified train/test sets using quit as the target."""
    if "quit" not in df.columns:
        raise ValueError("Expected target column 'quit' in the dataset.")
    x = df.loc[:, df.columns != "quit"]
    y = df["quit"]
    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
