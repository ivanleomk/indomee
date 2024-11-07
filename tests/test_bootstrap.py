from indomee import bootstrap_sample, calculate_metrics_at_k, bootstrap
import pandas as pd


def test_bootstrap_sample():
    preds = [
        ["a", "b"],
        ["c", "d"],
        ["e", "f"],
    ]
    labels = [["a", "b"], ["c", "d"], ["e", "f"]]

    result = bootstrap_sample(preds, labels, 10, ["recall"], [1, 2, 3])
    assert len(result.samples) == 10

    # Calculate metrics manually from the samples
    manual_results = []
    for sample in result.samples:
        manual_results.append(
            calculate_metrics_at_k(
                metrics=["recall"], preds=sample.pred, labels=sample.gt, k=[1, 2, 3]
            )
        )

    # Convert to mean metrics
    manual_metrics = pd.DataFrame(manual_results).mean().to_dict()

    print(manual_metrics)
    print(result.sample_metrics)

    for metric in manual_metrics:
        print(metric, manual_metrics[metric], result.sample_metrics[metric].value)
        assert manual_metrics[metric] == result.sample_metrics[metric].value


def test_bootstrap_samples():
    preds = [
        ["a", "b"],
        ["c", "d"],
        ["e", "f"],
    ]
    labels = [["a", "b"], ["c", "d"], ["e", "f"]]

    result = bootstrap(preds, labels, 10, 10, ["recall"], [1, 2, 3])

    # Calculate metrics manually from all samples
    manual_results = []
    for sample in result.samples:
        sample_metrics = {}
        for metric in sample.sample_metrics:
            sample_metrics[metric] = sample.sample_metrics[metric].value
        manual_results.append(sample_metrics)

    # Convert to mean metrics with confidence intervals
    df = pd.DataFrame(manual_results)
    manual_metrics = {}
    for col in df.columns:
        manual_metrics[col] = {
            "value": df[col].mean(),
            "ci_lower": df[col].quantile(0.025),
            "ci_upper": df[col].quantile(0.975),
        }

    # Compare manual calculations with result metrics
    for metric in result.sample_metrics:
        assert manual_metrics[metric]["value"] == result.sample_metrics[metric].value
        assert (
            manual_metrics[metric]["ci_lower"] == result.sample_metrics[metric].ci_lower
        )
        assert (
            manual_metrics[metric]["ci_upper"] == result.sample_metrics[metric].ci_upper
        )
