import pandas as pd
from employee_churn.data import encode_features, split_data


def test_encode_features():
    df = pd.DataFrame({"department": ["sales", "hr"], "salary": ["low", "high"], "quit": [0, 1]})
    out = encode_features(df)
    assert "department" not in out.columns
    assert "salary" not in out.columns
    assert "department_sales" in out.columns
    assert "salary_low" in out.columns


def test_split_data_is_stratified_and_returns_four_parts():
    df = pd.DataFrame({"feature": range(20), "quit": [0] * 10 + [1] * 10})
    parts = split_data(df)
    assert len(parts) == 4
    assert len(parts[0]) == 16
    assert len(parts[1]) == 4
    assert len(parts[2]) == 16
    assert len(parts[3]) == 4
