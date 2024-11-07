from indomee import bootstrap_sample, bootstrap
from rich import print

# Bootstrapping a sample
result = bootstrap_sample(
    preds=[["a", "b"], ["c", "d"], ["e", "f"]],
    labels=[["a", "b"], ["c", "d"], ["e", "f"]],
    sample_size=10,
    metrics=["recall"],
    k=[1, 2, 3],
)
print("Bootstrap Sample Metrics:", result.sample_metrics)
# {
#     "recall@1": BootstrapMetric(
#         name="recall@1",
#         value=np.float64(0.5),
#         ci_lower=np.float64(0.5),
#         ci_upper=np.float64(0.5),
#     ),
#     "recall@2": BootstrapMetric(
#         name="recall@2",
#         value=np.float64(1.0),
#         ci_lower=np.float64(1.0),
#         ci_upper=np.float64(1.0),
#     ),
#     "recall@3": BootstrapMetric(
#         name="recall@3",
#         value=np.float64(1.0),
#         ci_lower=np.float64(1.0),
#         ci_upper=np.float64(1.0),
#     ),
# }


# Bootstrapping multiple samples
result = bootstrap(
    preds=[["a", "b"], ["c", "d"], ["e", "f"]],
    labels=[["a", "b"], ["c", "d"], ["e", "f"]],
    n_samples=10,
    sample_size=2,
    metrics=["recall"],
    k=[1, 2, 3],
)
print("Bootstrap Metrics:", result.sample_metrics)
# {
#     "recall@1": BootstrapMetric(
#         name="recall@1",
#         value=np.float64(0.5),
#         ci_lower=np.float64(0.5),
#         ci_upper=np.float64(0.5),
#     ),
#     "recall@2": BootstrapMetric(
#         name="recall@2",
#         value=np.float64(1.0),
#         ci_lower=np.float64(1.0),
#         ci_upper=np.float64(1.0),
#     ),
#     "recall@3": BootstrapMetric(
#         name="recall@3",
#         value=np.float64(1.0),
#         ci_lower=np.float64(1.0),
#         ci_upper=np.float64(1.0),
#     ),
# }
