# DeepParameters Changelog

All notable changes to DeepParameters are documented here.

---

## [2.0.9] — March 2026

### Bug Fixes

- **`LSTMCPDLearner.get_cpd()` root-node crash** (`architectures.py`): When TensorFlow is absent and sklearn falls back to `MLPClassifier`, calling `predict()` on a root node returned a scalar `numpy.int64`, causing `TypeError: object of type 'numpy.int64' has no len()`. Fixed by using `predict_proba()` for sklearn models and checking `hasattr(pred, '__len__')` before `len(pred)`.
- **Stray DEBUG prints removed from `LSTMCPDLearner.get_cpd()`** (`architectures.py`): Four stray `print(f"DEBUG - Creating TabularCPD...")` lines were firing unconditionally during every LSTM CPD call. Removed.

### Code Quality

- **Linter false-positives suppressed** (`architectures.py`): Added `# type: ignore[import]` to optional TensorFlow imports inside `try/except` to silence Pylance/mypy unresolved-import errors in environments without TensorFlow installed.

### Testing

- All 8 neural-network architectures now pass: `test_all_deepparameters.py` — **8/8 architectures, 6/6 sampling methods (100%)**.

---

## [2.0.8] — March 2026

### Release Notes

Maintenance release. Version bump to resolve a Test PyPI conflict with the v2.0.7 artefact; no new code changes relative to v2.0.7.

---

## [2.0.7] — March 2026

### New Features

- **Command-line interface (CLI)**: A full terminal interface is now installed alongside the package. Two entry points are registered — `deepparameters` and the short alias `dp`. Commands:
  - `deepparameters info` — print version, all 9 architectures, and all 8 sampling methods
  - `deepparameters learn` — learn a CPD for one node from a CSV file
  - `deepparameters learn-network` — learn CPDs for all nodes in a network in parallel
  - `deepparameters compare` — compare a learned CPD to a ground-truth CPD and print all 7 metrics
- **CPD serialization format**: All CLI commands accept and produce a documented JSON format for `TabularCPD` objects (`variable`, `variable_card`, `values`, `evidence`, `evidence_card`, `state_names`).
- **`deepparameters/cli.py` module**: New module containing all CLI logic, importable as `from deepparameters.cli import main` or invoked via the binary.
- **`get_sampler()` accepts int or str keys**: Passing `sampling_method=4` (int) is now equivalent to `'4'` (str), eliminating a common `TypeError` for callers using integer sampling method IDs.

### Bug Fixes

- **Debug print removed from `format_cpd_output()`** (`utils.py`): A stray `print(f"DEBUG - CPD values shape: ...")` was firing unconditionally during every call, polluting user output. Removed.
- **NumPy 2.x compatibility** (`pyproject.toml`): The `<2.0.0` upper bound on numpy has been removed. The package is confirmed to contain no deprecated NumPy 1.x type aliases. Constraint is now `numpy>=1.19.0`.
- **LSTM root-node fallback** (`architectures.py`): `LSTMCPDLearner` now detects when a node has no parents. A `UserWarning` is emitted and `_build_model()` substitutes a simple dense network, since LSTM recurrence adds no value for root nodes.
- **`get_sampler()` int key support** (`sampling.py`): `str(sampling_method)` normalisation added; both `get_sampler(4)` and `get_sampler('4')` now work correctly.

### Documentation

- Updated `docs/API_REFERENCE.md` — added Section 9 "CLI Reference" covering all four commands with full flag tables and usage examples.
- Updated `docs/WORKFLOW_GUIDE.md` — added Section 11 "Using the CLI" with real-world usage patterns and JSON roundtrip examples.
- Updated `docs/README.md` — added CLI quick-start section.
- Updated `deepparameters_package/README.md` (PyPI) — added CLI section with quick-start and command reference.
- Updated `pyproject.toml` — added `[project.scripts]` block registering `deepparameters` and `dp` entry points.
- Updated `docs/04_todo_and_development_tracker.md` — marked all v2.0.7 sprint items complete; added completed milestone row.

### Testing

- Added `test_cli.py` (12 test groups, 64 assertions) — all tested via `subprocess.run` against the real CLI binary. **Result: 64 PASS / 0 FAIL**.

---

## [2.0.6] — 2026

### Bug Fixes

- **Label shift in `_prepare_data()`**: Target column and all parent columns are now encoded to contiguous `[0, n)` integer indices via the new `_encode_column()` static method. Previously, raw label values were passed directly to TensorFlow/sklearn, causing index-out-of-bounds errors for negative integers, non-contiguous integers, string labels, and boolean labels.

- **`_get_cpd_evidence()` recursion**: The helper in `utils.py` was calling itself (infinite recursion). Fixed to read parent variables directly from `cpd.variables[1:]`.

- **`_get_cpd_evidence_card()` array truth-value error**: Returned a numpy array, which caused `if not evidence_card` to raise `ValueError: ambiguous truth value`. Fixed to return a plain `list`.

- **`compare_cpds()` Frobenius norm crash on 3D arrays**: `np.linalg.norm(diff, 'fro')` does not support 3D arrays (CPDs with 2+ parents). Changed to `np.linalg.norm(diff.flatten())`.

- **`evaluate_cpd_performance` not exported**: The documented public alias for `compare_cpds` was not in `__init__.py` or `__all__`. Added.

- **`optimizer` and `early_stopping_patience` not forwarded**: These parameters were present in the documented API but not wired from `learn_cpd_for_node()` to the network learner classes. Fixed.

### New Features

- **`evaluate_cpd_performance` alias**: Public alias for `compare_cpds`. Import as `from deepparameters import evaluate_cpd_performance`. Returns 7 metrics: `mean_absolute_error`, `mean_squared_error`, `max_absolute_error`, `total_variation_distance`, `kl_divergence`, `frobenius_norm`, `cosine_similarity`.

- **`optimizer` parameter in `learn_cpd_for_node()`**: Accepts `'adam'`, `'adamw'`, `'sgd'`, `'rmsprop'`, `'nadam'`. Default: `'adam'`.

- **`early_stopping_patience` parameter in `learn_cpd_for_node()`**: Controls how many epochs without improvement before early stopping triggers. Default: `10`.

- **Automatic label encoding**: All discrete label types (strings, negative integers, non-contiguous integers, booleans) are automatically encoded to `[0, n)` before training. Original state values are preserved in `cpd.state_names`.

### Documentation

- Added `docs/API_REFERENCE.md` — complete, verified API reference for all public functions.
- Added `docs/WORKFLOW_GUIDE.md` — step-by-step practical guide with worked examples.
- Added `docs/CHANGELOG.md` — this file.
- Updated `docs/02_architecture.md` — added `optimizer` and `early_stopping_patience` to parameter table.
- Updated `docs/04_todo_and_development_tracker.md` — marked completed items.
- Updated root `README.md` — version bump, fixed pgmpy import pattern, added `evaluate_cpd_performance` section.

---

## [2.0.5] — 2025

### Features

- Nine network architectures: `simple`, `advanced`, `ultra`, `mega`, `lstm`, `bnn`, `vae`, `autoencoder`, `normalizing_flow`
- Eight CPD refinement sampling methods: Gibbs, Metropolis-Hastings, Importance Sampling, BPE, Variational Inference, HMC, SMC, Adaptive KDE
- Parallel network learning via `DeepParametersLearner`: `topological` and `parent_child` parallelism strategies
- Sklearn fallback when TensorFlow is not installed — all architectures remain usable
- `compare_cpds()` utility with 7 metrics
- `print_comparison_metrics()` — formatted console output
- `visualize_cpd()` — matplotlib/seaborn heatmap rendering
- `learn_network_parameters_parallel()` — standalone parallel learning function
- MLflow experiment tracking integration
- TestPyPI and PyPI publication

### Focus

- Initial multi-architecture, multi-sampling-method deep learning framework for Bayesian network CPD refinement
- Modular architecture (separate `architectures.py`, `sampling.py`, `core.py`, `utils.py`)

---

## [2.0.0] — 2024

### Initial Release

- Core `learn_cpd_for_node()` function
- Simple MLP-based CPD learning (sklearn)
- pgmpy integration for Bayesian network structure handling
- Basic experimental framework

---

## Version Policy

DeepParameters uses semantic versioning (`MAJOR.MINOR.PATCH`):
- **MAJOR**: Breaking API changes
- **MINOR**: Backward-compatible new features
- **PATCH**: Bug fixes and documentation updates
