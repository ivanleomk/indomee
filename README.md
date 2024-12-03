# Indomee

**Indomee** is a Python package designed to simplify the evaluation of retrieval-augmented generation (RAG) models and other retrieval-based systems. With `indomee`, you can compute common evaluation metrics like **recall** and **mean reciprocal rank (MRR)** at various levels of _k_, all through a straightforward API.

We also provide support for simple bootstrapping at the moment with t-testing coming soon.

## Installation

```bash
pip install indomee
```

You can get started with `indomee` with the following example

Indomee provides functions to calculate various metrics such as Mean Reciprocal Rank (MRR) and Recall.

#### Example Usage

```python
from indomee import calculate_mrr, calculate_recall, calculate_metrics_at_k

mrr = calculate_mrr([1, 2, 3], [2, 3, 4])
print("MRR:", mrr)
# > MRR: 0.5

# Calculate Recall
recall = calculate_recall([1, 2, 3], [2])
print("Recall:", recall)
# > Recall: 1

# Calculate metrics at specific k values
metrics = calculate_metrics_at_k(
    metrics=["recall"], preds=[1, 2, 3], labels=[2], k=[1, 2, 3]
)
print("Metrics at k:", metrics)
# > {'recall@1': 0.0, 'recall@2': 1.0, 'recall@3': 1.0}
```

### 2. Bootstrapping

Indomee also supports bootstrapping for more robust metric evaluation.

#### Example Usage

```python
from indomee import bootstrap_sample, bootstrap

# Bootstrapping a sample
result = bootstrap_sample(preds=[["a", "b"], ["c", "d"], ["e", "f"]], labels=[["a", "b"], ["c", "d"], ["e", "f"]], n_samples=10, metrics=["recall"], k=[1, 2, 3])
print("Bootstrap Sample Metrics:", result.sample_metrics)

# Bootstrapping multiple samples
result = bootstrap(preds=[["a", "b"], ["c", "d"], ["e", "f"]], labels=[["a", "b"], ["c", "d"], ["e", "f"]], n_samples=10, n_iterations=10, metrics=["recall"], k=[1, 2, 3])
print("Bootstrap Metrics:", result.sample_metrics)
```

### 3. T-Testing

For the last portion, we'll show how to perform a t-test between two different results that we've obtained from the different methods.

```python
from indomee import perform_t_tests
import pandas as pd

df = pd.read_csv("./data.csv")

# Calculate the mean for each method
method_1 = df["method_1"].tolist()
method_2 = df["method_2"].tolist()
baseline = df["baseline"].tolist()

results = perform_t_tests(
    baseline, method_1, method_2,
    names=["Baseline", "Method 1", "Method 2"],
    paired=True,
)
results
```
