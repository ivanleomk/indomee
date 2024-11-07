from typing import Literal


def is_valid_type(preds: list[str], labels: list[str]):
    type_pred = type(preds[0])

    for item in preds:
        assert isinstance(
            item, type_pred
        ), "predictions must be a list of the same type"

    for item in labels:
        assert isinstance(
            item, type_pred
        ), f"gt must be a list of the same type. gt is of type {type(labels)} while predictions is of type {type(preds)}"


def calculate_mrr(preds: list[str], labels: list[str]):
    is_valid_type(preds, labels)
    mrr = 0
    for label in labels:
        if label in preds:
            # Find the relevant item that has the smallest index
            mrr = max(mrr, 1 / (preds.index(label) + 1))
    return mrr


def calculate_recall(preds: list[str], labels: list[str]):
    is_valid_type(preds, labels)
    # Calculate the proportion of relevant items that were retrieved
    return len([label for label in labels if label in preds]) / len(labels)


def calculate_metrics_at_k(
    metrics: Literal["recall", "mrr"],
    preds: list[str],
    labels: list[str],
    k: list[int],
):
    for metric in metrics:
        if metric not in ["recall", "mrr"]:
            raise ValueError(
                f"Invalid metric: {metric} - we only support recall and mrr for now"
            )

    results = {}

    for metric in metrics:
        if metric == "recall":
            for subset_k in k:
                results[f"{metric}@{subset_k}"] = calculate_recall(
                    preds[:subset_k], labels
                )
        elif metric == "mrr":
            for subset_k in k:
                results[f"{metric}@{subset_k}"] = calculate_mrr(
                    preds[:subset_k], labels
                )

    return results
