# DeepParameters Documentation

**Updated documentation for the `deepparameters` Python package — v2.0.9.**

This folder is your reference point for understanding, using, and contributing to developing the package. Start here.

---

## Documents in This Folder

### [API Reference](API_REFERENCE.md)
*Complete, accurate API documentation for every public function, class, and CLI command.*

Covers every parameter of `learn_cpd_for_node()`, `DeepParametersLearner`, `evaluate_cpd_performance`, and parallel functions — with runnable code examples verified against v2.0.9. Also includes Section 13: the full CLI reference for `deepparameters info`, `learn`, `learn-network`, and `compare`.

**Start here if you want to look up a specific function, parameter, or CLI flag.**

---

### [Workflow Guide](WORKFLOW_GUIDE.md)
*Step-by-step guide: from raw data to a fully-fitted Bayesian network.*

Covers the complete workflow: data preparation → CPD learning → evaluation → inference. Includes full working examples (medical diagnosis, financial risk, multi-architecture comparison). Also covers the CLI workflow in Section 11.

**Start here if you want to learn how to use the package end-to-end.**

---

### [Changelog](CHANGELOG.md)
*What changed in each version.*

---

### [01 — Theory & Concepts](01_theory_and_concepts.md)
*What the package is actually doing, explained for non-experts.*

Covers:
- What Bayesian Networks are (with a simple everyday example)
- What a Conditional Probability Distribution (CPD) is and why it matters
- Why classical statistics (Maximum Likelihood Estimation) struggles with sparse data
- How DeepParameters uses neural networks to solve this problem
- What "sampling" means in this context (it's not what you might think)
- All 12 refinement methods explained in plain English
- Parallel learning for whole networks
- Performance metrics (MAE, RMSE, KL divergence, etc.)
- A full worked example with real code and predicted numbers

**Read this first if you're new to the package or to Bayesian networks.**

---

### [02 — Architecture](02_architecture.md)
*How the code is organised and how the pieces connect.*

Covers:
- Repository structure (`core.py`, `architectures.py`, `sampling.py`, etc.)
- The 9 neural network architectures — what each does and when to use it
- The class hierarchy (`BaseCPDLearner`, `BaseSampler`, and all subclasses)
- The factory pattern (`_get_network_learner()` and `get_sampler()`)
- The parallel learning engine (`ParallelCPDLearner`, `FactorGroup`, decomposition styles)
- A step-by-step data flow diagram through the entire pipeline
- Complete API reference for all public functions
- Dependency table and Python version requirements
- TensorFlow vs sklearn fallback — what you lose without TF and how it works

**Read this when you want to understand the internals or contribute code.**

---

### [03 — Deployment Roadmap](03_deployment_roadmap.md)
*Where the package came from and where it's going.*

Covers:
- Current state (v2.0.9) — what's live on PyPI right now
- Full version history from v0.0.1 to v2.0.9
- Step-by-step deployment workflow (build → Test PyPI → production PyPI)
- What's planned for v2.1.x (automatic architecture selection, numpy 2.x compatibility)
- Known limitations of the current version

**Read this when planning a release or understanding the project's direction.**

---

### [04 — Development Tracker & TODO](04_todo_and_development_tracker.md)
*A living checklist of what needs to be done.*

Covers:
- Active sprint items for v2.0.9
- Bug fixes (with code location, description, and effort estimate)
- Technical debt items
- Feature backlog (near-term, medium-term, long-term)
- Testing gaps
- Documentation tasks
- API improvement ideas
- Research and academic tasks
- History of completed milestones

**Keep this document up to date. Add to it whenever you find a new issue.**

---

## Quick Reference

### Install

```bash
pip install deepparameters                  # sklearn architectures only
pip install deepparameters[tensorflow]      # full — all 9 architectures
```

### Basic Usage

```python
from pgmpy.models import DiscreteBayesianNetwork
from deepparameters import learn_cpd_for_node
import pandas as pd

# Your data — must be discretized (integer values)
data = pd.DataFrame({...})

# Your Bayesian network structure
bn = DiscreteBayesianNetwork([('A', 'B'), ('A', 'C'), ('B', 'D')])

# Learn CPD for one node
cpd = learn_cpd_for_node(
    node='D',
    data=data,
    true_model=bn,
    learnt_bn_structure=bn,
    num_parameters=10,
    network_type='simple',  # see 02_architecture.md for all 9 options
    sampling_method='4',    # see 01_theory_and_concepts.md for all 12 options
)

print(cpd)
```

### Learn All CPDs in Parallel

```python
from deepparameters import learn_network_parameters_parallel

cpds = learn_network_parameters_parallel(
    data=data,
    true_model=bn,
    learnt_bn_structure=bn,
    network_type='simple',
    sampling_method='4',
    parallel_style='topological',  # or 'parent_child'
)
# cpds is a dict: {'D': TabularCPD, 'C': TabularCPD, ...}
```

### Evaluate a Learned CPD

```python
from deepparameters import evaluate_cpd_performance

metrics = evaluate_cpd_performance(learned_cpd, true_cpd)
print(f"MAE: {metrics['mean_absolute_error']:.4f}")
print(f"KL:  {metrics['kl_divergence']:.4f}")
```

### Available Architectures (`network_type=`)

| Value | Description |
|-------|-------------|
| `'simple'` | MLP baseline — fast, always works |
| `'advanced'` | Deeper MLP |
| `'ultra'` | Even deeper MLP |
| `'mega'` | Maximum MLP capacity |
| `'lstm'` | LSTM (TF recommended) |
| `'bnn'` | Bayesian Neural Network with uncertainty |
| `'vae'` | Variational Autoencoder |
| `'autoencoder'` | Standard Autoencoder |
| `'normalizing_flow'` | Normalizing Flows |

### Available Sampling Methods (`sampling_method=`)

| Value | Method | Speed |
|-------|--------|-------|
| `'1'` | Gibbs Sampling (MCMC) | Slow |
| `'2'` | Metropolis-Hastings | Slow-Medium |
| `'3'` | Importance Sampling | Medium |
| `'4'` | Bayesian Parameter Estimation | Fast |
| `'5'` | Variational Inference | Medium |
| `'6'` | Hamiltonian Monte Carlo | Slow |
| `'7'` | Sequential Monte Carlo | Slow |
| `'8'` | Adaptive KDE Sampling | Medium |
| `'9'` | Weighted Sampling | Very Fast |
| `'10'` | Stratified Sampling | Fast |
| `'11'` | KDE Sampling | Medium |
| `'12'` | Dirichlet Bayesian Sampling | Medium |

---

## Package Links

- **PyPI:** https://pypi.org/project/deepparameters/
- **Source:** `deepparameters_package/` in this repository
- **Author:** Rudzani Mulaudzi, Wits University
- **License:** MIT
- **Current Version:** 2.0.9
