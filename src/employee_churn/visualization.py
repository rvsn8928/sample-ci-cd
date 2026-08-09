from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz


def plot_turnover_by_column(df, column, output_dir="outputs"):
    """Save a turnover-frequency bar chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd_crosstab = __import__("pandas").crosstab(df[column], df["quit"])
    ax = pd_crosstab.plot(kind="bar", figsize=(12, 8))
    ax.set_title(f"Turnover Frequency on {column.title()}")
    ax.set_xlabel(column.title())
    ax.set_ylabel("Frequency of turnover")
    fig = ax.get_figure()
    fig.tight_layout()
    path = output_dir / f"turnover_by_{column}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_tree(model, feature_names, output_path="outputs/decision_tree.png", class_names=None):
    """Export a fitted decision tree to a PNG via Graphviz."""
    from graphviz import Source

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot = export_graphviz(
        model,
        out_file=None,
        feature_names=list(feature_names),
        class_names=class_names or ["stayed", "quit"],
        filled=True,
    )
    Source(dot).render(str(output_path.with_suffix("")), format="png", cleanup=True)
    return output_path
