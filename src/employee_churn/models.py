from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(
    x_train, y_train, criterion="gini", splitter="best", max_depth=2,
    min_samples_split=2, min_samples_leaf=1, random_state=0
):
    model = DecisionTreeClassifier(
        random_state=random_state,
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    return model.fit(x_train, y_train)


def train_random_forest(
    x_train, y_train, criterion="gini", bootstrap=True, n_estimators=100,
    max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=1
):
    model = RandomForestClassifier(
        random_state=random_state,
        criterion=criterion,
        bootstrap=bootstrap,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
    )
    return model.fit(x_train, y_train)


def evaluate_model(model, x_train, y_train, x_test, y_test) -> Dict:
    """Return the same core accuracy evaluation used in the notebook."""
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    metrics = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "classification_report": classification_report(y_test, test_pred, output_dict=True),
    }
    if hasattr(model, "predict_proba"):
        try:
            metrics["test_roc_auc"] = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        except ValueError:
            pass
    return metrics
