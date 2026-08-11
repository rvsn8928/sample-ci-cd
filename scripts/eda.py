import argparse
from employee_churn.data import encode_features, load_data
from employee_churn.visualization import plot_turnover_by_column


def main():
    parser = argparse.ArgumentParser(description="Run employee churn EDA.")
    parser.add_argument("--data", default="data/employee_data.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    df = load_data(args.data)
    print(df.head())
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nTarget distribution:\n", df["quit"].value_counts())
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nTarget distribution:\n", df["quit"].value_counts())

    for column in ("salary", "department"):
        if column in df.columns:
            path = plot_turnover_by_column(df, column, args.output)
            print("Saved:", path)

    encoded = encode_features(df)
    print("\nEncoded columns:", list(encoded.columns))


if __name__ == "__main__":
    main()
