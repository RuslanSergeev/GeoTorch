import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List


def categorize_data(
    data: pd.DataFrame,
    category_column: str,
    num_categories: int,
    output_column: str,
):
    data = data.copy()
    cat_min = data[category_column].min()
    cat_max = data[category_column].max()
    threshold = (cat_max - cat_min) / num_categories / 10
    bins = np.linspace(cat_min - threshold, cat_max + threshold, num_categories + 1)
    labels = [(left+right)/2 for left, right in zip(bins[:-1], bins[1:])]
    data[output_column] = pd.cut(data[category_column], bins=bins, labels=labels, include_lowest=True)
    return data, labels


def get_CM(data: pd.DataFrame):
    return confusion_matrix(data["damage_true"], data["damage"])


def get_metrics(
    data: pd.DataFrame,
    category_column: str,
    num_categories: int,
) -> pd.DataFrame:
    out_column = category_column + "_categorized"
    data, labels = categorize_data(data, category_column, num_categories, out_column)
    metrics = []
    for label in labels:
        try:
            sub_data = data.loc[data[out_column] == label]
            TN, FP, FN, TP = get_CM(sub_data).ravel()
            if TP + FN != 0:
                sensitivity = TP / (TP + FN)
            else:
                sensitivity = 0
            if TN + FP != 0:
                specificity = TN / (TN + FP)
            else:
                specificity = 0
            balanced_accuracy = (sensitivity + specificity) / 2
            metrics.append({
                category_column: label,
                "num_samples": len(sub_data),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
            })
        except Exception as e:
            print(f"Error: {e}, {label=}")
    return pd.DataFrame(metrics)


def get_correlation(
    data: pd.DataFrame,
    category_column: str,
    num_categories: List[int],
):
    correlation = []
    for num_category in num_categories:
        metrics = get_metrics(data, category_column, num_category)
        corr_matrix = metrics[[category_column, "sensitivity", "specificity", "balanced_accuracy"]].corr()
        correlation.append(
            {
                "num_categories": num_category,
                f"{category_column} * sensitivity": corr_matrix.loc[category_column, "sensitivity"],
                f"{category_column} * specificity": corr_matrix.loc[category_column, "specificity"],
                f"{category_column} * balanced_accuracy": corr_matrix.loc[category_column, "balanced_accuracy"],
            }
        )
    return pd.DataFrame(correlation)

