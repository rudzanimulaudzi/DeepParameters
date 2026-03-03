# DeepParameters Workflow Guide

**Version 2.0.9 — Practical step-by-step guide to using DeepParameters.**

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Data Preparation](#2-data-preparation)
3. [Define Your Network Structure](#3-define-your-network-structure)
4. [Learn a Single CPD](#4-learn-a-single-cpd)
5. [Advanced Configuration](#5-advanced-configuration)
6. [Learn CPDs for an Entire Network](#6-learn-cpds-for-an-entire-network)
7. [Evaluate and Compare CPDs](#7-evaluate-and-compare-cpds)
8. [Architecture Comparison](#8-architecture-comparison)
9. [Complete Worked Examples](#9-complete-worked-examples)
10. [Troubleshooting](#10-troubleshooting)
11. [Using the CLI](#11-using-the-cli)

---

## 1. Installation & Setup

```bash
pip install deepparameters

# Optional: TensorFlow for LSTM, BNN, VAE, Autoencoder, Normalizing Flow
pip install deepparameters[tensorflow]
```

**Verify the install:**

```python
import deepparameters
print(deepparameters.__version__)  # 2.0.9

from deepparameters import learn_cpd_for_node, DeepParametersLearner
from deepparameters import evaluate_cpd_performance, print_comparison_metrics
```

---

## 2. Data Preparation

DeepParameters takes a `pd.DataFrame` where each column is a **discrete variable** and each row is one sample. All values must be reducible to comparable discrete categories.

### Minimum Requirements

- At least 30 samples per state combination (preferably 100+)
- No `NaN` values
- Each column must have at least 2 unique values
- All values within a column must be comparable (no mixing ints and strings)

### Example: Medical data discretization

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

# Raw data might have continuous columns — discretize them
np.random.seed(42)
raw_data = pd.DataFrame({
    'age':        np.random.normal(50, 15, 500),       # continuous
    'cholesterol': np.random.normal(200, 40, 500),     # continuous
    'smoker':     np.random.choice([0, 1], 500),       # already discrete
    'heart_disease': np.random.choice([0, 1], 500),   # already discrete
})

# Discretize continuous columns
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
raw_data['age']        = discretizer.fit_transform(raw_data[['age']]).astype(int)
raw_data['cholesterol'] = discretizer.fit_transform(raw_data[['cholesterol']]).astype(int)

print(raw_data.head())
print(raw_data.dtypes)
# All columns must be int-compatible discrete values
```

### Handling Labels Automatically

DeepParameters handles any discrete label encoding automatically, including:
- **Strings**: `'low'`, `'medium'`, `'high'` → automatically encoded
- **Negative integers**: `-1, 0, 1` → automatically encoded
- **Non-contiguous integers**: `2, 5, 8` → automatically encoded

You don't need to manually remap labels. The original values are preserved in `cpd.state_names`.

---

## 3. Define Your Network Structure

Use pgmpy to specify the Bayesian network structure as a directed acyclic graph (DAG).

```python
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

# Define structure as a list of (parent, child) edge tuples
structure = BayesianNetwork([
    ('age',    'heart_disease'),
    ('smoker', 'heart_disease'),
    ('cholesterol', 'heart_disease'),
])

# You can also learn structure from data using pgmpy
from pgmpy.estimators import HillClimbSearch, BicScore

hc = HillClimbSearch(raw_data)
learnt_structure = hc.estimate(scoring_method=BicScore(raw_data))
```

---

## 4. Learn a Single CPD

```python
from deepparameters import learn_cpd_for_node

cpd = learn_cpd_for_node(
    node='heart_disease',
    data=raw_data,
    true_model=structure,         # the "true" or reference model
    learnt_bn_structure=structure, # the model being estimated
    num_parameters=10,
)
print(cpd)
```

**What happens internally:**
1. Parent columns for `heart_disease` (`age`, `smoker`, `cholesterol`) are extracted from `data`
2. Labels are encoded to `[0, n)` integer indices
3. A neural network (default: simple MLP) is trained on the encoded data
4. The network's output softmax probabilities form the CPD entries
5. A `TabularCPD` is built with the original label names preserved in `state_names`

---

## 5. Advanced Configuration

### Choosing a Network Architecture

```python
# For small datasets or quick testing
cpd_simple = learn_cpd_for_node('heart_disease', raw_data, structure, structure,
                                 num_parameters=10, network_type='simple')

# For more expressive modelling
cpd_advanced = learn_cpd_for_node('heart_disease', raw_data, structure, structure,
                                   num_parameters=50, network_type='advanced')

# For uncertainty quantification (uses Bayesian neural network with MC Dropout)
cpd_bnn = learn_cpd_for_node('heart_disease', raw_data, structure, structure,
                               num_parameters=30, network_type='bnn')

# For latent-space generative modelling
cpd_vae = learn_cpd_for_node('heart_disease', raw_data, structure, structure,
                               num_parameters=40, network_type='vae')
```

Available architectures: `'simple'`, `'advanced'`, `'ultra'`, `'mega'`, `'lstm'`, `'bnn'`, `'vae'`, `'autoencoder'`, `'normalizing_flow'`

### Choosing a Sampling Method

```python
# Default — Bayesian Parameter Estimation (fast, data-driven)
cpd_bpe = learn_cpd_for_node('heart_disease', raw_data, structure, structure,
                              num_parameters=10, sampling_method='4')

# Highest accuracy — Metropolis-Hastings MCMC
cpd_mh = learn_cpd_for_node('heart_disease', raw_data, structure, structure,
                              num_parameters=10, sampling_method='2')

# Sparse data — Adaptive KDE
cpd_kde = learn_cpd_for_node('heart_disease', raw_data, structure, structure,
                              num_parameters=10, sampling_method='8')
```

Always pass `sampling_method` as a **string** (`'1'`, `'2'`, ..., `'8'`).

### Controlling Training

```python
cpd = learn_cpd_for_node(
    node='heart_disease',
    data=raw_data,
    true_model=structure,
    learnt_bn_structure=structure,
    num_parameters=50,
    network_type='vae',
    sampling_method='5',            # Variational Inference
    epochs=300,
    batch_size=64,
    learning_rate=0.0005,
    validation_split=0.2,
    early_stopping=True,
    optimizer='adamw',              # 'adam', 'adamw', 'sgd', 'rmsprop', 'nadam'
    early_stopping_patience=20,     # wait 20 epochs before stopping
    verbose=True,
    random_state=42,
)
```

---

## 6. Learn CPDs for an Entire Network

Use `DeepParametersLearner` when you want to learn CPDs for all nodes in the network, potentially in parallel.

### Topological Parallelism

Groups nodes by their depth level in the DAG and processes each level in parallel (nodes in the same level have no dependencies on each other).

```python
from deepparameters import DeepParametersLearner

learner = DeepParametersLearner()

cpds = learner.learn_network_parallel(
    data=raw_data,
    network_structure=structure,
    parallel_style='topological',   # group by DAG depth level
    max_workers=4,                  # processes in the pool
    epochs=100,
    network_type='simple',
    verbose=False,
)

print(f"Learned {len(cpds)} CPDs")
for cpd in cpds:
    print(f"  {cpd.variable}: {cpd.variable_card} states, {len(cpd.variables)-1} parents")
```

### Parent-Child Parallelism

Groups nodes by their immediate family relationships. Can be faster for hierarchical networks.

```python
cpds = learner.learn_network_parallel(
    data=raw_data,
    network_structure=structure,
    parallel_style='parent_child',
    max_workers=4,
    epochs=100,
    max_time_per_group=60,  # stop a group after 60 seconds (optional)
    verbose=False,
)
```

### Adding Learned CPDs to a pgmpy Model

```python
# Create a new model with the learned CPDs
learned_model = BayesianNetwork(structure.edges())
learned_model.add_cpds(*cpds)
print(f"Model valid: {learned_model.check_model()}")
```

### Standalone Parallel Function

```python
from deepparameters import learn_network_parameters_parallel

cpds = learn_network_parameters_parallel(
    data=raw_data,
    true_model=structure,
    learnt_bn_structure=structure,
    network_type='advanced',
    sampling_method='4',
    parallel_style='topological',
    max_workers=4,
)
```

---

## 7. Evaluate and Compare CPDs

After learning, evaluate how close your learned CPD is to the true CPD using multiple metrics.

### Compare a Learned CPD to the True CPD

```python
from deepparameters import evaluate_cpd_performance, print_comparison_metrics
from pgmpy.factors.discrete import TabularCPD

# Define the ground-truth CPD
true_cpd = TabularCPD(
    variable='heart_disease', variable_card=2,
    values=[[0.9, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.1, 0.3, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]],
    evidence=['age', 'smoker', 'cholesterol'],
    evidence_card=[3, 2, 3],            # must match num states from data
)

# Learn the CPD
learned_cpd = learn_cpd_for_node(
    node='heart_disease', data=raw_data,
    true_model=structure, learnt_bn_structure=structure,
    num_parameters=20, network_type='advanced',
)

# Evaluate
metrics = evaluate_cpd_performance(learned_cpd, true_cpd)
print_comparison_metrics(metrics)
```

### Programmatic Access to Metrics

```python
metrics = evaluate_cpd_performance(learned_cpd, true_cpd)

# All seven metrics:
print(f"MAE:  {metrics['mean_absolute_error']:.4f}")
print(f"MSE:  {metrics['mean_squared_error']:.4f}")
print(f"Max:  {metrics['max_absolute_error']:.4f}")
print(f"TVD:  {metrics['total_variation_distance']:.4f}")
print(f"KL:   {metrics['kl_divergence']:.4f}")
print(f"Frob: {metrics['frobenius_norm']:.4f}")
print(f"Cos:  {metrics['cosine_similarity']:.4f}")
```

---

## 8. Architecture Comparison

Compare multiple network architectures on the same node:

```python
architectures = ['simple', 'advanced', 'ultra', 'vae', 'bnn']
results = {}

for arch in architectures:
    try:
        cpd = learn_cpd_for_node(
            node='heart_disease',
            data=raw_data,
            true_model=structure,
            learnt_bn_structure=structure,
            num_parameters=20,
            network_type=arch,
            epochs=100,
            verbose=False,
        )
        metrics = evaluate_cpd_performance(cpd, true_cpd)
        results[arch] = {
            'mae': metrics['mean_absolute_error'],
            'kl':  metrics['kl_divergence'],
        }
        print(f"{arch:20s} | MAE: {metrics['mean_absolute_error']:.4f} | KL: {metrics['kl_divergence']:.4f}")
    except Exception as e:
        print(f"{arch:20s} | Error: {e}")

# Find best architecture by MAE
best_arch = min(results, key=lambda k: results[k]['mae'])
print(f"\nBest architecture: {best_arch} (MAE = {results[best_arch]['mae']:.4f})")
```

---

## 9. Complete Worked Examples

### Example 1: Medical Diagnosis (VAE Architecture)

```python
import pandas as pd
import numpy as np

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from deepparameters import learn_cpd_for_node, evaluate_cpd_performance

np.random.seed(42)
n = 500

# Simulate medical data
data = pd.DataFrame({
    'age':           np.random.choice([0, 1, 2], n),       # young, middle, old
    'family_history': np.random.choice([0, 1], n),
    'smoker':         np.random.choice([0, 1], n),
    'diagnosis':      np.random.choice([0, 1, 2], n),      # healthy, at-risk, sick
})

structure = BayesianNetwork([
    ('age',            'diagnosis'),
    ('family_history', 'diagnosis'),
    ('smoker',         'diagnosis'),
])

cpd = learn_cpd_for_node(
    node='diagnosis',
    data=data,
    true_model=structure,
    learnt_bn_structure=structure,
    num_parameters=40,
    network_type='vae',
    sampling_method='5',
    epochs=150,
    optimizer='adam',
    early_stopping=True,
    early_stopping_patience=15,
    verbose=False,
    random_state=42,
)
print(cpd)
print(f"\nCPD shape: {cpd.values.shape}")
print(f"Parents: {cpd.variables[1:]}")
```

---

### Example 2: Financial Risk (BNN with Uncertainty)

```python
import pandas as pd
import numpy as np

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from deepparameters import learn_cpd_for_node

np.random.seed(0)
n = 400

data = pd.DataFrame({
    'market_condition': np.random.choice([0, 1, 2], n),   # bear, neutral, bull
    'credit_rating':    np.random.choice([0, 1, 2], n),   # poor, ok, good
    'leverage':         np.random.choice([0, 1], n),       # low, high
    'default_risk':     np.random.choice([0, 1, 2], n),   # low, medium, high
})

structure = BayesianNetwork([
    ('market_condition', 'default_risk'),
    ('credit_rating',    'default_risk'),
    ('leverage',         'default_risk'),
])

cpd = learn_cpd_for_node(
    node='default_risk',
    data=data,
    true_model=structure,
    learnt_bn_structure=structure,
    num_parameters=30,
    network_type='bnn',        # Bayesian Neural Network — provides uncertainty
    sampling_method='6',       # Hamiltonian Monte Carlo
    epochs=200,
    batch_size=32,
    optimizer='adamw',
    early_stopping=True,
    early_stopping_patience=20,
    verbose=False,
    random_state=0,
)
print(cpd)
```

---

### Example 3: Multi-Architecture Benchmarking

```python
import pandas as pd
import numpy as np
import time

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from deepparameters import learn_cpd_for_node, evaluate_cpd_performance

np.random.seed(42)
n = 300

data = pd.DataFrame({
    'A': np.random.choice([0, 1], n),
    'C': np.random.choice([0, 1], n),
    'B': np.random.choice([0, 1], n),
})
model = BayesianNetwork([('A', 'B'), ('C', 'B')])

true_cpd = TabularCPD(
    variable='B', variable_card=2,
    values=[[0.8, 0.4, 0.6, 0.2],
            [0.2, 0.6, 0.4, 0.8]],
    evidence=['A', 'C'], evidence_card=[2, 2],
)

architectures = ['simple', 'advanced', 'ultra', 'bnn', 'vae']
print(f"{'Architecture':<20} | {'MAE':>7} | {'KL':>7} | {'Time(s)':>8}")
print("-" * 55)

for arch in architectures:
    t0 = time.time()
    try:
        cpd = learn_cpd_for_node(
            node='B', data=data, true_model=model,
            learnt_bn_structure=model, num_parameters=25,
            network_type=arch, sampling_method='4',
            epochs=100, verbose=False, random_state=42,
        )
        metrics = evaluate_cpd_performance(cpd, true_cpd)
        elapsed = time.time() - t0
        print(f"{arch:<20} | {metrics['mean_absolute_error']:7.4f} | "
              f"{metrics['kl_divergence']:7.4f} | {elapsed:8.1f}s")
    except Exception as e:
        print(f"{arch:<20} | Error: {e}")
```

---

## 10. Troubleshooting

### "node 'X' not found in data"

```python
# Check your column names match exactly
print(data.columns.tolist())  # must include the node name
```

### Model gives poor accuracy

- Increase `epochs` (try 200–500)
- Increase `num_parameters` (try 50–100)
- Try a more expressive architecture: `'advanced'`, `'vae'`, or `'bnn'`
- Try `sampling_method='2'` (Metropolis-Hastings) for highest accuracy
- Ensure you have enough training data (aim for 100+ samples per state combination)

### TensorFlow architectures using sklearn fallback

This is normal when TensorFlow is not installed. To enable TF:

```bash
pip install tensorflow
# or, on Apple Silicon:
pip install tensorflow-macos tensorflow-metal
```

### NaN or mixed-type errors

```python
# Check for NaN
print(data.isnull().sum())

# Fill NaN
data = data.fillna(data.mode().iloc[0])

# Check types
print(data.dtypes)

# Check for mixed-type columns
for col in data.columns:
    types = data[col].apply(type).unique()
    if len(types) > 1:
        print(f"Column '{col}' has mixed types: {types}")
```

### `evaluate_cpd_performance` raises ValueError

Both CPDs must be for the same node, with the same parents and cardinalities. The error message will indicate what differs.

---

## 11. Using the CLI

DeepParameters v2.0.9 ships with a terminal interface. After installation, two entry points are available: `deepparameters` and the short alias `dp`. All commands work identically with either name.

### Check the install

```bash
deepparameters info
# Output: version, 9 architectures, 8 sampling methods
```

### Prepare your files

**data.csv** — one row per sample, one column per node (discrete values):

```
A,B,C
0,1,0
1,0,1
0,0,0
1,1,1
```

**edges.csv** — two-column file defining the DAG edges (header required):

```
parent,child
A,B
C,B
```

### Learn a single CPD

```bash
# Print CPD JSON to stdout
deepparameters learn --node B --data data.csv --edges edges.csv --num-parameters 10

# Save to file with advanced options
deepparameters learn \
  --node B \
  --data data.csv \
  --edges edges.csv \
  --num-parameters 20 \
  --network-type vae \
  --sampling-method 4 \
  --epochs 200 \
  --optimizer adamw \
  --early-stopping-patience 15 \
  --output cpd_B.json
```

### Learn all CPDs in parallel

```bash
deepparameters learn-network \
  --data data.csv \
  --edges edges.csv \
  --parallel-style topological \
  --max-workers 4 \
  --output-dir ./learned_cpds/
# Creates: learned_cpds/cpd_A.json, learned_cpds/cpd_B.json, ...
```

### Compare learned vs. ground-truth

```bash
deepparameters compare \
  --learned cpd_B_learned.json \
  --true    cpd_B_true.json \
  --output  metrics_B.json
```

**Example output:**

```
Mean Absolute Error:     0.0521
Mean Squared Error:      0.0043
Root Mean Square Error:  0.0654
Max Absolute Error:      0.1200
KL Divergence:           0.0189
Cosine Similarity:       0.9923
Frobenius Norm:          0.0921
```

### JSON roundtrip with Python

CPD JSON files produced by the CLI can be loaded back into Python:

```python
import json
import numpy as np
from pgmpy.factors.discrete import TabularCPD

with open("cpd_B.json") as f:
    d = json.load(f)

cpd = TabularCPD(
    variable=d["variable"],
    variable_card=d["variable_card"],
    values=np.array(d["values"]),
    evidence=d["evidence"] or None,
    evidence_card=d["evidence_card"] or None,
    state_names=d.get("state_names"),
)
print(cpd)
```

### Full CLI reference

See [docs/API_REFERENCE.md — Section 13](API_REFERENCE.md#13-cli-reference) for complete flag tables and output format documentation.
