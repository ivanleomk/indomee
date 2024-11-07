from dataclasses import dataclass
import random
from indomee.metrics import calculate_metrics_at_k
from typing import Literal
import pandas as pd


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
