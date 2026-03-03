# DeepParameters API Reference

**Version 2.0.9 — All examples verified against the actual package.**

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Import Reference](#2-quick-import-reference)
3. [`learn_cpd_for_node()`](#3-learn_cpd_for_node)
4. [`DeepParametersLearner`](#4-deepparameterslearner)
5. [`evaluate_cpd_performance()` / `compare_cpds()`](#5-evaluate_cpd_performance--compare_cpds)
6. [`print_comparison_metrics()`](#6-print_comparison_metrics)
7. [`visualize_cpd()`](#7-visualize_cpd)
8. [`learn_network_parameters_parallel()`](#8-learn_network_parameters_parallel)
9. [Network Type Reference](#9-network-type-reference)
10. [Sampling Method Reference](#10-sampling-method-reference)
11. [Error Reference](#11-error-reference)
12. [Data Requirements](#12-data-requirements)
13. [CLI Reference](#13-cli-reference)

---

## 1. Installation

```bash
# Base install — uses sklearn for all architectures
pip install deepparameters

# Full install — enables TensorFlow-based architectures (lstm, bnn, vae, etc.)
pip install deepparameters[tensorflow]

# Development install from source
git clone https://github.com/rudzanimulaudzi/DeepParameters.git
cd DeepParameters/deepparameters_package
pip install -e .
```

**Python:** 3.8+  
**Required:** numpy, pandas, scikit-learn, pgmpy, networkx, scipy, matplotlib, seaborn  
**Optional:** tensorflow (for full architecture support)

---

## 2. Quick Import Reference

```python
# Primary interface
from deepparameters import learn_cpd_for_node

# Object-oriented interface (for whole-network learning)
from deepparameters import DeepParametersLearner
from deepparameters.core import DeepParametersLearner  # equivalent

# Evaluation
from deepparameters import evaluate_cpd_performance   # recommended
from deepparameters import compare_cpds               # same function, legacy name

# Utilities
from deepparameters import print_comparison_metrics, visualize_cpd

# Parallel learning (standalone function)
from deepparameters import learn_network_parameters_parallel

# pgmpy (recommended import pattern for compatibility)
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from pgmpy.factors.discrete import TabularCPD
```

---

## 3. `learn_cpd_for_node()`

The main entry point. Trains a neural network on your data and returns a `TabularCPD` for one node.

### Signature

```python
learn_cpd_for_node(
    node: str,
    data: pd.DataFrame,
    true_model: BayesianNetwork,
    learnt_bn_structure: BayesianNetwork,
    num_parameters: int,
    network_type: str = "simple",
    sampling_method: str = "1",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    early_stopping: bool = True,
    optimizer: str = "adam",
    early_stopping_patience: int = 10,
    verbose: bool = False,
    random_state: int = 42,
) -> TabularCPD
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node` | `str` | *required* | Name of the node whose CPD to learn. Must be a column in `data` and a node in `learnt_bn_structure`. |
| `data` | `pd.DataFrame` | *required* | Training data. All columns must be **discrete integers** (see [Data Requirements](#12-data-requirements)). |
| `true_model` | `BayesianNetwork` | *required* | The true Bayesian network structure. Used internally for some evaluation steps. |
| `learnt_bn_structure` | `BayesianNetwork` | *required* | The learned network structure. Defines which columns are parents of `node`. |
| `num_parameters` | `int` | *required* | Controls the neural network size. Higher = more capacity. Typical range: 5–100. |
| `network_type` | `str` | `"simple"` | Architecture. See [Network Type Reference](#9-network-type-reference). |
| `sampling_method` | `str` | `"1"` | CPD refinement method. See [Sampling Method Reference](#10-sampling-method-reference). |
| `epochs` | `int` | `100` | Training epochs. |
| `batch_size` | `int` | `32` | Mini-batch size for training. |
| `learning_rate` | `float` | `0.001` | Learning rate for the optimizer. |
| `validation_split` | `float` | `0.2` | Fraction of training data withheld for validation (used by early stopping). |
| `early_stopping` | `bool` | `True` | Whether to stop training early when validation loss plateaus. |
| `optimizer` | `str` | `"adam"` | Optimizer. Options: `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`, `"nadam"`. |
| `early_stopping_patience` | `int` | `10` | Epochs to wait with no improvement before stopping (only used when `early_stopping=True`). |
| `verbose` | `bool` | `False` | Print training progress and CPD details. |
| `random_state` | `int` | `42` | Seed for reproducibility. |

### Returns

`TabularCPD` — a pgmpy `TabularCPD` object. Properties:
- `.variable` — the node name
- `.variable_card` — number of states
- `.values` — the probability array (shape: `[node_card, *parent_cards]`)
- `.state_names` — dict mapping variable names to their original state values
- `.evidence` / `.evidence_card` — parent variables and their cardinalities

### Examples

**Basic usage:**

```python
import pandas as pd
import numpy as np
from deepparameters import learn_cpd_for_node

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.choice([0, 1], 300),
    'C': np.random.choice([0, 1], 300),
    'B': np.random.choice([0, 1], 300),
})
true_model   = BayesianNetwork([('A', 'B'), ('C', 'B')])
learnt_model = BayesianNetwork([('A', 'B'), ('C', 'B')])

cpd = learn_cpd_for_node(
    node='B',
    data=data,
    true_model=true_model,
    learnt_bn_structure=learnt_model,
    num_parameters=10,
)
print(cpd)
```

**Advanced configuration:**

```python
cpd = learn_cpd_for_node(
    node='B',
    data=data,
    true_model=true_model,
    learnt_bn_structure=learnt_model,
    num_parameters=50,
    network_type='vae',
    sampling_method='8',
    epochs=200,
    batch_size=64,
    learning_rate=0.001,
    validation_split=0.2,
    early_stopping=True,
    optimizer='adamw',
    early_stopping_patience=20,
    verbose=True,
    random_state=42,
)
```

**With non-standard label values (strings, negative integers, non-contiguous):**

```python
# DeepParameters handles any discrete values automatically.
# Labels are encoded to [0, n) internally; original values
# are preserved in cpd.state_names.
data_str = pd.DataFrame({
    'Weather': np.random.choice(['sunny', 'cloudy', 'rainy'], 200),
    'Mood':    np.random.choice(['happy', 'sad', 'neutral'], 200),
})
model = BayesianNetwork([('Weather', 'Mood')])
cpd = learn_cpd_for_node('Mood', data_str, model, model, num_parameters=10)
print(cpd.state_names)  # {'Mood': ['happy', 'neutral', 'sad'], 'Weather': [...]}
```

### Exceptions

| Exception | Cause |
|-----------|-------|
| `ValueError` | `node` not in `data.columns`, invalid `network_type`, data contains NaN, parents not found |
| `TypeError` | Column contains mixed or incomparable types |
| `KeyError` | A required column is missing from `data` |

---

## 4. `DeepParametersLearner`

An object-oriented interface for learning CPDs across an entire network.

### Import

```python
from deepparameters import DeepParametersLearner
# or
from deepparameters.core import DeepParametersLearner
```

### `learn_network_parallel()`

```python
learner = DeepParametersLearner()
cpds = learner.learn_network_parallel(
    data: pd.DataFrame,
    network_structure: BayesianNetwork,
    parallel_style: str = "topological",   # or "parent_child"
    max_workers: int = None,               # defaults to CPU count
    max_time_per_group: float = None,      # seconds; None = no limit
    network_type: str = "simple",
    sampling_method: str = "1",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping: bool = True,
    verbose: bool = False,
) -> list[TabularCPD]
```

**Example:**

```python
from deepparameters import DeepParametersLearner
import pandas as pd
import numpy as np

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

np.random.seed(0)
data = pd.DataFrame({
    'A': np.random.choice([0, 1], 500),
    'C': np.random.choice([0, 1], 500),
    'B': np.random.choice([0, 1], 500),
})
bn = BayesianNetwork([('A', 'B'), ('C', 'B')])

learner = DeepParametersLearner()

# Topological — nodes grouped by depth in the DAG
cpds = learner.learn_network_parallel(
    data=data,
    network_structure=bn,
    parallel_style='topological',
    max_workers=4,
    epochs=100,
    verbose=False,
)

# Parent-child — nodes grouped by family clusters (faster for hierarchical networks)
cpds = learner.learn_network_parallel(
    data=data,
    network_structure=bn,
    parallel_style='parent_child',
    max_workers=4,
    epochs=100,
    network_type='advanced',
    max_time_per_group=60,
    verbose=False,
)
```

### `benchmark_parallel_performance()`

```python
results = learner.benchmark_parallel_performance(
    data: pd.DataFrame,
    network_structure: BayesianNetwork,
    parallel_style: str = "parent_child",
    max_workers_list: list[int] = [1, 2, 4],
    epochs: int = 20,
    network_type: str = "simple",
)
# Prints a timing table and returns a dict with timing results
```

**Example:**

```python
results = learner.benchmark_parallel_performance(
    data=data,
    network_structure=bn,
    parallel_style='parent_child',
    max_workers_list=[1, 2, 4],
    epochs=20,
)
```

---

## 5. `evaluate_cpd_performance()` / `compare_cpds()`

Compare a learned CPD to the true CPD and get quantitative metrics.

`evaluate_cpd_performance` and `compare_cpds` are the **same function** — `evaluate_cpd_performance` is the documented public name.

### Signature

```python
evaluate_cpd_performance(
    learned_cpd: TabularCPD,
    true_cpd: TabularCPD,
) -> dict
```

### Returns

A `dict` with the following keys:

| Key | Description | Perfect Value |
|-----|-------------|---------------|
| `mean_absolute_error` | Average |learned - true| across all CPD entries | `0.0` |
| `mean_squared_error` | Average squared error | `0.0` |
| `max_absolute_error` | Worst single entry error | `0.0` |
| `total_variation_distance` | `0.5 * sum(|learned - true|)` | `0.0` |
| `kl_divergence` | KL(true ‖ learned) — information-theoretic distance | `0.0` |
| `frobenius_norm` | Frobenius norm of the difference | `0.0` |
| `cosine_similarity` | Cosine similarity of flattened arrays | `1.0` |

### Example

```python
from deepparameters import evaluate_cpd_performance, learn_cpd_for_node
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
import numpy as np

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.choice([0, 1], 300),
    'C': np.random.choice([0, 1], 300),
    'B': np.random.choice([0, 1], 300),
})
model = BayesianNetwork([('A', 'B'), ('C', 'B')])

# True CPD
true_cpd = TabularCPD(
    variable='B', variable_card=2,
    values=[[0.8, 0.4, 0.6, 0.2],
            [0.2, 0.6, 0.4, 0.8]],
    evidence=['A', 'C'],
    evidence_card=[2, 2],
)

# Learned CPD
learned_cpd = learn_cpd_for_node(
    node='B', data=data, true_model=model,
    learnt_bn_structure=model, num_parameters=10,
)

# Compare
metrics = evaluate_cpd_performance(learned_cpd, true_cpd)
print(f"MAE:  {metrics['mean_absolute_error']:.4f}")
print(f"KL:   {metrics['kl_divergence']:.4f}")
print(f"Max:  {metrics['max_absolute_error']:.4f}")
```

### Raises

`ValueError` if the two CPDs have different structures (different node, cardinality, parents, or parent cardinalities). Both CPDs must be for the same node.

---

## 6. `print_comparison_metrics()`

Pretty-prints the metrics dict returned by `evaluate_cpd_performance()`.

```python
from deepparameters import print_comparison_metrics

metrics = evaluate_cpd_performance(learned_cpd, true_cpd)
print_comparison_metrics(metrics)
```

Output:
```
CPD COMPARISON METRICS
========================================
Mean Absolute Error:      0.042300
Mean Squared Error:       0.003100
Max Absolute Error:       0.109000
Total Variation Distance: 0.169200
KL Divergence:            0.007800
Frobenius Norm:           0.062000
Cosine Similarity:        0.987000
```

---

## 7. `visualize_cpd()`

Renders a heatmap of a `TabularCPD` using matplotlib and seaborn.

```python
from deepparameters import visualize_cpd

visualize_cpd(cpd, title="Learned CPD for node B")
```

Requires matplotlib and seaborn (both installed automatically with the package).

---

## 8. `learn_network_parameters_parallel()`

Standalone function version of `DeepParametersLearner.learn_network_parallel()`.

```python
from deepparameters import learn_network_parameters_parallel

cpds = learn_network_parameters_parallel(
    data=data,
    true_model=bn,
    learnt_bn_structure=bn,
    network_type='simple',
    sampling_method='4',
    parallel_style='topological',
    max_workers=4,
)
```

---

## 9. Network Type Reference

Pass as `network_type=` to `learn_cpd_for_node()`.

| Value | Architecture | TF Needed? | Best For |
|-------|-------------|-----------|----------|
| `'simple'` | MLP (64, 32) | No | Fast baseline, always works |
| `'advanced'` | MLP (128, 64, 32) | No | General purpose |
| `'ultra'` | MLP (256, 128, 64, 32) | No | Higher accuracy |
| `'mega'` | MLP (512, 256, 128, 64, 32) | No | Maximum MLP capacity |
| `'lstm'` | LSTM (64→32) + Dense | Yes* | Temporal/sequential patterns |
| `'bnn'` | Dense + MC Dropout | Yes* | Uncertainty quantification |
| `'vae'` | Encoder→Latent→Decoder | Yes* | Generative/probabilistic |
| `'autoencoder'` | Encoder→Bottleneck→Decoder | Yes* | Feature compression |
| `'normalizing_flow'` | Affine coupling layers | Yes* | Exact density modelling |

\* Falls back to sklearn MLP if TensorFlow is not installed. The API is identical either way.

---

## 10. Sampling Method Reference

Pass as `sampling_method=` to `learn_cpd_for_node()`. Always a **string**, not an integer.

| Value | Method | Speed | Best For |
|-------|--------|-------|----------|
| `'1'` | Gibbs Sampling | Slow | Full MCMC consistency with network |
| `'2'` | Metropolis-Hastings | Slow | Broadest probability space exploration |
| `'3'` | Importance Sampling | Medium | Rare-event probability estimation |
| `'4'` | Bayesian Parameter Estimation (BPE) | Fast | Data-driven, interpretable. Good default. |
| `'5'` | Variational Inference | Medium | Large networks where MCMC is too slow |
| `'6'` | Hamiltonian Monte Carlo | Slow | High-accuracy MCMC with gradient guidance |
| `'7'` | Sequential Monte Carlo | Slow | Streaming data, sequential updates |
| `'8'` | Adaptive KDE | Medium | Sparse data, smooth density interpolation |

**Quick guidance:**
- Default/fastest → `'4'` (BPE)
- Highest accuracy → `'2'` (Metropolis-Hastings) or `'6'` (HMC)
- Sparse data → `'8'` (Adaptive KDE)
- Uncertainty-aware → `'5'` (Variational Inference)

---

## 11. Error Reference

| Error | Common Cause | Fix |
|-------|-------------|-----|
| `ValueError: node 'X' not found` | `node` string doesn't match a column in `data` | Check column names |
| `ValueError: Column 'X' contains NaN` | Missing values in data | Drop or impute: `data.fillna(data.mode().iloc[0])` |
| `TypeError: Column 'X' mixed types` | Column has both int and str values | Clean column before passing |
| `KeyError: 'X'` | Parent column missing from `data` | Ensure all nodes in `learnt_bn_structure` have corresponding columns |
| `ValueError: CPDs have different structures` | Comparing CPDs for different nodes in `evaluate_cpd_performance` | Ensure both CPDs are for the same variable |
| `ValueError: Unsupported network type: ...` | Typo in `network_type` string | See [Network Type Reference](#9-network-type-reference) |

---

## 12. Data Requirements

DeepParameters requires **discrete integer data**. All values must be non-negative integers representing categories.

### What is supported

```python
# ✅ Zero-based integers (most common)
pd.DataFrame({'A': [0, 1, 2], 'B': [0, 1, 0]})

# ✅ Non-contiguous integers (encoded automatically)
pd.DataFrame({'A': [2, 5, 8], 'B': [2, 5, 8]})

# ✅ Negative integers (encoded automatically)
pd.DataFrame({'A': [-1, 0, 1], 'B': [-1, 0, 1]})

# ✅ Strings (encoded automatically to sorted integer indices)
pd.DataFrame({'A': ['low', 'medium', 'high'], 'B': ['yes', 'no', 'yes']})

# ✅ Boolean (treated as 0/1)
pd.DataFrame({'A': [True, False, True], 'B': [False, True, True]})
```

### What is NOT supported

```python
# ❌ Continuous floats — must discretize first
pd.DataFrame({'A': [1.5, 2.3, 0.8]})

# ❌ NaN values — must fill or drop
pd.DataFrame({'A': [0, None, 1]})

# ❌ Mixed types in one column
pd.DataFrame({'A': [0, 'one', 2]})
```

### Discretizing continuous data

```python
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
data['income_disc'] = discretizer.fit_transform(data[['income']]).astype(int)
```

---

## 13. CLI Reference

DeepParameters ships with a terminal interface. Two entry points are registered: `deepparameters` and the short alias `dp`.

### Installation & verification

```bash
pip install deepparameters          # entry points installed automatically
deepparameters --version            # check version
deepparameters --help               # list all commands
dp --help                           # same, using the short alias
```

### `deepparameters info`

Print the installed version, all 9 architectures, and all 8 sampling methods.

```bash
deepparameters info
```

No flags required.

---

### `deepparameters learn`

Learn the CPD for a single node from a CSV file and a network structure file.

```bash
deepparameters learn \
  --node heart_disease \
  --data data.csv \
  --edges edges.csv \
  --num-parameters 20 \
  --output cpd_heart_disease.json
```

**Flags**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--node` | ✅ | — | Name of the target node (must match a column in the CSV) |
| `--data` | ✅ | — | Path to CSV with discrete data |
| `--edges` | ✅ | — | Path to two-column CSV (`parent,child`) defining model structure |
| `--num-parameters` | ✅ | — | Neural network capacity (integer) |
| `--output` | No | stdout | Path to write the CPD JSON; omit to print to stdout |
| `--network-type` | No | `simple` | Architecture: `simple`, `advanced`, `ultra`, `mega`, `lstm`, `bnn`, `vae`, `autoencoder`, `normalizing_flow` |
| `--sampling-method` | No | `4` | Sampling method ID: `1`–`8` |
| `--epochs` | No | `100` | Training epochs |
| `--batch-size` | No | `32` | Mini-batch size |
| `--learning-rate` | No | `0.001` | Learning rate |
| `--optimizer` | No | `adam` | Optimizer: `adam`, `adamw`, `sgd`, `rmsprop`, `nadam` |
| `--early-stopping-patience` | No | `10` | Epochs without improvement before stopping |
| `--no-early-stopping` | No | off | Flag to disable early stopping entirely |
| `--verbose` | No | off | Enable verbose training output |
| `--random-state` | No | `42` | Random seed |

**edges.csv format** (header row required):

```
parent,child
A,B
C,B
```

**Output CPD JSON format:**

```json
{
  "variable": "B",
  "variable_card": 2,
  "values": [[0.8, 0.4], [0.2, 0.6]],
  "evidence": ["A"],
  "evidence_card": [2],
  "state_names": {"B": [0, 1], "A": [0, 1]}
}
```

---

### `deepparameters learn-network`

Learn CPDs for all nodes in a Bayesian network in parallel.

```bash
deepparameters learn-network \
  --data data.csv \
  --edges edges.csv \
  --parallel-style topological \
  --max-workers 4 \
  --output-dir ./cpds/
```

**Flags**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data` | ✅ | — | Path to CSV |
| `--edges` | ✅ | — | Path to edges CSV |
| `--output-dir` | ✅ | — | Directory to write one JSON file per node |
| `--parallel-style` | No | `topological` | `topological` or `parent_child` |
| `--max-workers` | No | `4` | Thread pool size |
| `--network-type` | No | `simple` | Architecture (same options as `learn`) |
| `--sampling-method` | No | `4` | Sampling method ID (`1`–`8`) |
| `--epochs` | No | `100` | Training epochs per node |

Output files are named `cpd_<NodeName>.json` inside `--output-dir`.

---

### `deepparameters compare`

Compare a learned CPD (JSON file) against a ground-truth CPD (JSON file) and print all 7 metrics.

```bash
deepparameters compare \
  --learned cpd_learned.json \
  --true    cpd_true.json \
  --output  metrics.json
```

**Flags**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--learned` | ✅ | — | Path to JSON file containing the learned CPD |
| `--true` | ✅ | — | Path to JSON file containing the ground-truth CPD |
| `--output` | No | stdout | Path to write the metrics as JSON; omit to print only |

**Printed metrics:**

```
Mean Absolute Error:       0.0521
Mean Squared Error:        0.0043
Root Mean Square Error:    0.0654
Max Absolute Error:        0.1200
KL Divergence:             0.0189
Cosine Similarity:         0.9923
Frobenius Norm:            0.0921
```

The `--output` JSON has keys: `mean_absolute_error`, `mean_squared_error`, `max_absolute_error`, `kl_divergence`, `frobenius_norm`, `cosine_similarity`, `total_variation_distance`.

---

### Full end-to-end CLI workflow

```bash
# 1. Inspect available architectures
deepparameters info

# 2. Learn one CPD
deepparameters learn \
  --node heart_disease \
  --data medical_data.csv \
  --edges network.csv \
  --num-parameters 20 \
  --network-type vae \
  --sampling-method 4 \
  --epochs 200 \
  --optimizer adamw \
  --output cpd_heart.json

# 3. Learn all CPDs in parallel
deepparameters learn-network \
  --data medical_data.csv \
  --edges network.csv \
  --parallel-style topological \
  --max-workers 4 \
  --output-dir ./learned_cpds/

# 4. Compare learned CPD against ground truth
deepparameters compare \
  --learned cpd_heart.json \
  --true    cpd_heart_true.json \
  --output  heart_metrics.json
```
