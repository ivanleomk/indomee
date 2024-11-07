from indomee import calculate_mrr, calculate_recall, calculate_metrics_at_k
import pytest


def test_mrr():
    assert calculate_mrr([1, 2, 3], [2, 3, 4]) == 0.5


def test_recall():
    assert calculate_recall([1, 2, 3], [2]) == 1
    assert calculate_recall([1, 2, 3], [4, 5, 6]) == 0
    assert calculate_recall([1, 2, 3], [1, 2, 3]) == 1
    assert round(calculate_recall([1, 2, 3], [1, 2, 4]), 2) == 0.67


def test_calculate_metrics_at_k():
    assert calculate_metrics_at_k(
        metrics=["recall"], preds=[1, 2, 3], labels=[2], k=[1, 2, 3]
    ) == {
        "recall@1": 0,
        "recall@2": 1,
        "recall@3": 1,
    }
    assert calculate_metrics_at_k(
        metrics=["mrr"], preds=[1, 2, 3], labels=[2], k=[1, 2, 3]
    ) == {
        "mrr@1": 0,
        "mrr@2": 0.5,
        "mrr@3": 0.5,
    }


def test_throw_error_on_invalid_metric():
    with pytest.raises(ValueError):
        calculate_metrics_at_k(["invalid"], [1, 2, 3], [2], [1, 2, 3])


def test_throw_error_on_invalid_types():
    with pytest.raises(AssertionError):
        calculate_metrics_at_k(["recall"], [1, 2, 3], ["2"], [1, 2, 3])

    with pytest.raises(AssertionError):
        calculate_metrics_at_k(["mrr"], [1, 2, 3], ["2"], [1, 2, 3])
