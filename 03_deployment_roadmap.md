# DeepParameters: Deployment Roadmap

**Where the package is, where it came from, and where it's going.**

---

## Table of Contents

1. [Current State — v2.0.9](#1-current-state--v209)
2. [Version History](#2-version-history)
3. [Deployment Workflow (How to Publish to PyPI)](#3-deployment-workflow-how-to-publish-to-pypi)
4. [What's Next — v2.1.x (Planned)](#4-whats-next--v21x-planned)
5. [Longer-Term Roadmap](#5-longer-term-roadmap)
6. [Known Limitations & Debt](#6-known-limitations--debt)

---

## 1. Current State — v2.0.9

**Live on PyPI:** https://pypi.org/project/deepparameters/

| Detail | Value |
|--------|-------|
| Version | 2.0.9 |
| Status | Beta |
| License | MIT |
| Python | 3.8+ |
| Author | Rudzani Mulaudzi (Wits University) |
| Published | Production PyPI ✅ |

### What v2.0.9 Includes

- **9 neural network architectures**: simple, advanced, ultra, mega, lstm, bnn, vae, autoencoder, normalizing_flow
- **12 statistical refinement methods** (sampling methods `'1'` through `'12'`)
- **2 parallel learning styles**: topological and parent-child
- **Full sklearn fallback** when TensorFlow is not installed
- **Clean public API**: `learn_cpd_for_node()` and `learn_network_parameters_parallel()`
- **Utility functions**: `compare_cpds()`, `visualize_cpd()`, `print_comparison_metrics()`
- **7 performance metrics**: MAE, RMSE, KL divergence, JS divergence, cosine similarity, max error, consistency check
- **CLI entry points**: `deepparameters` and `dp` (commands: `info`, `learn`, `learn-network`, `compare`)

### Install v2.0.9 Now

```bash
pip install deepparameters            # base (sklearn architectures only)
pip install deepparameters[tensorflow] # full (all 9 architectures)
```

---

## 2. Version History

### The Early Prototype Period (v0.0.x)

These were early experimental releases — proof-of-concept stages while the API and approach were being figured out.

| Version | Key Milestone |
|---------|--------------|
| 0.0.1 | Initial PyPI release — basic CPD learning concept |
| 0.0.2 | Improvements to package structure |
| 0.0.3 | Additional fixes |
| 0.0.4 | Stability improvements |
| 0.0.5 | Further refinements |
| 0.0.6 | Final prototype — laid groundwork for v2.0.x restructure |

### The Production Period (v2.0.x)

A **major rewrite and restructuring** — not a continuation of the 0.0.x numbering but a deliberate signal that this was a new, more serious release.

| Version | Release | Key Changes |
|---------|---------|-------------|
| **2.0.0** | Early 2025 | Complete architectural rewrite. New module structure (`core.py`, `architectures.py`, `sampling.py`, `utils.py`). All 9 architectures. Clean `learn_cpd_for_node()` API. |
| **2.0.1** | 2025 | Bug fixes from v2.0.0. Refinements to the sampling pipeline. |
| **2.0.2** | 2025 | Stability improvements. Better error messages. |
| **2.0.3** | 2025 | Added `parallel.py` — initial parallel learning engine (topological style only). Full test suite validation. |
| **2.0.5** | Jan 2025 | **Parent-child parallel style** added (`parallel_style='parent_child'`). 20–40% speed improvement for hierarchical networks. Advanced benchmarking. Dual decomposition strategy. |
| **2.0.6** | 2025 | Bug fixes. PEP 621 compliant packaging. Improved import system. TF optional dependency handling. |
| **2.0.7** | Mar 2026 | CLI (`deepparameters` / `dp`). 4 bug fixes: debug print removed, NumPy 2.x support, LSTM root fallback, `get_sampler` int keys. |
| **2.0.8** | Mar 2026 | Maintenance bump; 2 further fixes: LSTM root-node sklearn crash (`numpy.int64 has no len()`), stray DEBUG prints in `LSTMCPDLearner.get_cpd()`. |
| **2.0.9** | Mar 2026 | Linter false-positives suppressed (`# type: ignore`) for optional TF imports. All 8 architectures 100% pass. Production PyPI release. |

> **Note:** Version 2.0.4 does not appear in the PyPI history — it may have been a build that was skipped or uploaded only briefly to Test PyPI.

---

## 3. Deployment Workflow (How to Publish to PyPI)

This section explains how to go from code changes to a live PyPI release. The full steps apply each time a new version is published.

### Prerequisites

```bash
pip install build twine
```

### Step 1: Bump the Version Number

Update **both** of these files to the new version:

1. `deepparameters_package/pyproject.toml`:
   ```toml
   [project]
   version = "2.0.9"
   ```

2. `deepparameters_package/deepparameters/__init__.py`:
   ```python
   __version__ = "2.0.9"
   ```

### Step 2: Build the Distributions

```bash
cd "/Users/rudzani/Downloads/Python_Projects/UV_Demo/deepparameters refinement/deepparameters_package"

# Remove old builds first
rm -rf dist/

# Build wheel + source tarball
python -m build
```

This produces:
- `dist/deepparameters-2.0.9.tar.gz` — source distribution
- `dist/deepparameters-2.0.9-py3-none-any.whl` — wheel

### Step 3: Deploy to Test PyPI (Always Do This First)

```bash
twine upload --repository testpypi dist/*
```

You will be prompted for your Test PyPI credentials. If using an API token:
- Username: `__token__`
- Password: `pypi-...` (your token)

### Step 4: Verify on Test PyPI

```bash
# Create a clean temporary environment
python3 -m venv /tmp/test_dp
source /tmp/test_dp/bin/activate

# Install from Test PyPI
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  deepparameters==2.0.9

# Run a quick smoke test
python -c "import deepparameters; print(deepparameters.__version__)"
```

The `test_testpypi_2_0_8.py` script in the workspace root automates this verification step.

### Step 5: Deploy to Production PyPI

Only after Test PyPI verification passes:

```bash
twine upload dist/*
```

### Step 6: Verify Production Install

```bash
pip install deepparameters==2.0.9
python -c "import deepparameters; print(deepparameters.__version__)"
```

### Version Number Conventions

- Patch (2.0.x → 2.0.x+1): Bug fixes, no API changes
- Minor (2.x.0 → 2.x+1.0): New non-breaking features or new architectures/samplers
- Major (x.0.0 → x+1.0.0): Breaking API changes or fundamental architectural rewrites

---

## 4. What's Next — v2.1.x (Planned)

With v2.0.9 now on production PyPI, the next meaningful milestone is v2.1.x, which will introduce intelligence and automation features rather than purely maintenance changes.

### High-Priority Candidates

- [ ] **Automatic architecture recommendation** — inspect dataset size, parent count, and cardinality and suggest the best `network_type`
- [ ] **Automatic sampling method selection** — recommend `sampling_method` based on dataset characteristics
- [ ] **Type hints throughout** — add Python type annotations to all public functions for IDE support and readability
- [ ] **`__repr__` methods** on learner classes — makes debugging easier in notebooks
- [ ] **Drop Python 3.8** — 3.8 reached end-of-life Oct 2024; target 3.9+ going forward

### Stretch Goals for v2.1.x

- [ ] **`list_available_architectures()` returns a dict**, not just prints — allows programmatic access
- [ ] **Packaged unit tests** — move test suite into `deepparameters_package/tests/` for `pytest` integration
- [ ] **`__repr__` on `TabularCPD` wrappers** — cleaner REPL inspection

---

## 5. Longer-Term Roadmap

These were listed in `RELEASE_NOTES_v2.0.5.md` — medium-to-long-term vision items:

### v2.1.x — Intelligence & Automation

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Automatic parallel style selection** | An AI recommendation module that inspects the network structure and recommends 'topological' vs 'parent_child' | Medium |
| **Automatic architecture recommendation** | Suggest the best `network_type` based on dataset size, number of parents, and cardinality | Medium |
| **Automatic sampling method selection** | Recommend `sampling_method` based on dataset characteristics | Medium |

### v2.2.x — Performance

| Feature | Description | Complexity |
|---------|-------------|------------|
| **GPU acceleration** | Leverage GPU hardware for faster neural network training, especially for large networks | High |
| **Hybrid parallel approaches** | Combine topological and parent-child decompositions for maximum efficiency | High |
| **Adaptive batch sizing** | Automatically tune `batch_size` based on available memory | Medium |

### v3.x — Distributed & Scale

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Distributed learning** | Multi-machine parallel processing for very large Bayesian networks | Very High |
| **Streaming data support** | Learn CPDs incrementally as new data arrives | High |
| **Async API** | `asyncio`-compatible interface for non-blocking CPD learning in web applications | Medium |

### Long-Term Scientific Goals

- **Published paper** — the LaTeX methodology document (`DEEPPARAMETERS_FORMAL_METHODOLOGY.tex`) suggests an academic publication is planned
- **Benchmarking against classical methods** — formal comparison with pure pgmpy estimation, EM algorithm, and Bayesian parameter estimation
- **Real-world case studies** — applying DeepParameters to medical, financial, or engineering Bayesian networks with publicly available datasets

---

## 6. Known Limitations & Debt

These are known issues as of v2.0.9:

| Issue | Location | Severity | Notes |
|-------|----------|----------|-------|
| `_get_cpd_evidence()` recursion guard | `utils.py` | Medium | Should add depth limit |
| Python 3.8 EOL | `pyproject.toml` | Low | 3.8 reached end-of-life Oct 2024 |
| No type hints | All modules | Low | Makes IDE support weaker |
| No `__repr__` on learner classes | `architectures.py` | Low | Hard to inspect objects in notebooks |
| MCMC samplers use pgmpy's `GibbsSampling` | `sampling.py` | Medium | pgmpy's sampler may not always converge for all structures |
| No unit test suite in the package | `deepparameters_package/` | High | All tests are in root workspace scripts, not packaged tests |

---

*Last updated for version 2.0.9. Update this document whenever a version is released or planned features change.*
