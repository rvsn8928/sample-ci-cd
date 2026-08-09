import argparse
import json
from pathlib import Path

from employee_churn.data import encode_features, load_data, split_data
from employee_churn.models import evaluate_model, train_decision_tree, train_random_forest
from employee_churn.visualization import plot_turnover_by_column, save_tree


def main():
    parser = argparse.ArgumentParser(description="Train employee churn models.")
    parser.add_argument("--data", default="data/employee_data.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.data)
    encoded = encode_features(raw)
    x_train, x_test, y_train, y_test = split_data(encoded)

    for column in ("salary", "department"):
        if column in raw.columns:
            plot_turnover_by_column(raw, column, output)

    dt = train_decision_tree(x_train, y_train, max_depth=args.depth)
    rf = train_random_forest(x_train, y_train, max_depth=args.depth, n_estimators=args.trees)

    metrics = {
        "decision_tree": evaluate_model(dt, x_train, y_train, x_test, y_test),
        "random_forest": evaluate_model(rf, x_train, y_train, x_test, y_test),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    try:
        save_tree(dt, x_train.columns, output / "decision_tree.png")
        save_tree(rf.estimators_[0], x_train.columns, output / "random_forest_tree_0.png")
    except Exception as exc:
        print(f"Tree visualization skipped: {exc}")

    print(f"Decision Tree test accuracy: {metrics['decision_tree']['test_accuracy']:.3f}")
    print(f"Random Forest test accuracy: {metrics['random_forest']['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
