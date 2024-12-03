from dataclasses import dataclass
import random
from indomee.metrics import calculate_metrics_at_k
from typing import Literal, Sequence
import pandas as pd
from statistics import mean, stdev


@dataclass
class BootstrapMetric:
    name: str
    value: float
    ci_lower: float
    ci_upper: float


@dataclass
class Sample:
    """
    This is a single sample that we've randomly sampled from the predictions and its corresponding gt
    """

    pred: list[str]
    gt: list[str]


@dataclass
class BootstrapSample:
    """
    This is a class which returns a list of sampled predictions and ground truth pairs
    """

    samples: list[Sample]
    sample_metrics: dict[str, BootstrapMetric]


@dataclass
class BootstrapResult:
    samples: list[BootstrapSample]
    sample_metrics: dict[str, BootstrapMetric]


def metrics_from_df(df: pd.DataFrame) -> dict[str, BootstrapMetric]:
    bootstrap_metrics = {}
    for col in df.columns:
        bootstrap_metrics[col] = BootstrapMetric(
            name=col,
            value=df[col].mean(),
            ci_lower=df[col].quantile(0.025),
            ci_upper=df[col].quantile(0.975),
        )

    return bootstrap_metrics


def bootstrap_sample(
    preds: list[str],
    labels: list[str],
    sample_size: int,
    metrics: list[Literal["mrr", "recall"]],
    k: list[int],
):
    """
    This returns a single Bootstrap Sample Object
    """
    samples = []
    results = []
    for _ in range(sample_size):
        idx = random.randint(0, len(preds) - 1)
        samples.append(Sample(pred=preds[idx], gt=labels[idx]))
        results.append(
            calculate_metrics_at_k(
                metrics=metrics, preds=preds[idx], labels=labels[idx], k=k
            )
        )

    return BootstrapSample(
        samples=samples, sample_metrics=metrics_from_df(pd.DataFrame(results))
    )


def bootstrap(
    preds: list[list[str]],
    labels: list[list[str]],
    n_samples: int,
    sample_size: int,
    metrics: list[Literal["mrr", "recall"]],
    k: list[int],
):
    samples = []
    results = []
    for _ in range(n_samples):
        bootstrap_sample_result = bootstrap_sample(
            preds, labels, sample_size, metrics, k
        )
        samples.append(bootstrap_sample_result)
        results.append(
            {
                metric: bootstrap_sample_result.sample_metrics[metric].value
                for metric in bootstrap_sample_result.sample_metrics
            }
        )

    return BootstrapResult(
        samples=samples,
        sample_metrics=metrics_from_df(pd.DataFrame(results)),
    )


def bootstrap_from_results(
    results: Sequence[float],
    n_samples: int = 1000,
    sample_size: int | None = None,
    confidence_level: float = 0.95,
) -> BootstrapMetric:
    """
    Performs bootstrapping on raw metric results to calculate confidence intervals. We take the mean of the results and then calculate the confidence intervals from the bootstrap samples.

    Args:
        results: List of metric values to bootstrap from
        n_samples: Number of bootstrap samples to draw
        sample_size: Size of each bootstrap sample. If None, uses len(results)
        confidence_level: Confidence level for intervals (default 0.95 for 95% CI)

    Returns:
        BootstrapMetric containing the mean value and confidence intervals
    """
    if sample_size is None:
        sample_size = len(results)

    bootstrap_samples = []
    for _ in range(n_samples):
        # Draw random samples with replacement
        sample = [random.choice(results) for _ in range(sample_size)]
        bootstrap_samples.append(mean(sample))

    # Calculate confidence intervals
    sorted_samples = sorted(bootstrap_samples)
    lower_idx = int((1 - confidence_level) / 2 * n_samples)
    upper_idx = int((1 - (1 - confidence_level) / 2) * n_samples)

    return BootstrapMetric(
        name="bootstrap_metric",
        value=mean(results),
        ci_lower=sorted_samples[lower_idx],
        ci_upper=sorted_samples[upper_idx],
    )
