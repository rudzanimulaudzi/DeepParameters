# 🚀 Advanced Neural Network CPD Learning for Bayesian Networks

🏠 **Homepage**: [https://github.com/rudzanimulaudzi/DeepParameters](https://github.com/rudzanimulaudzi/DeepParameters)

DeepParameters is a comprehensive Python package for learning Conditional Probability Distributions (CPDs) using state-of-the-art neural network architectures. It provides a unified interface for experimenting with various deep learning approaches to probabilistic modeling.

## 🚀 Key Features

- **9 Neural Network Architectures**: Simple NN, Advanced NN, LSTM, Autoencoder, VAE, BNN, Normalizing Flow, Ultra, Mega
- **8 Sampling Methods**: Gibbs, Metropolis-Hastings, Importance, BPE, Variational, HMC, and more
- **Configurable Parallel Learning**: Choose between 'topological' and 'parent_child' parallel execution styles
- **Parallel CPD Learning**: Multi-threaded parameter learning with factor group decomposition
- **Comprehensive Evaluation**: 7 performance metrics including MAE, KL divergence, and probability consistency
- **Simple Interface**: Unified `learn_cpd_for_node()` function for all architectures

## 🔧 Data Preprocessing Requirements

**⚠️ IMPORTANT**: DeepParameters requires **discrete data** for proper functioning. All variables in your dataset must be categorical/discrete rather than continuous.

### Data Discretization Steps

Before using DeepParameters, ensure your data is properly discretized:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

# Example: Converting continuous data to discrete
def preprocess_data_for_deepparameters(data, continuous_columns=None, n_bins=3):
    """
    Prepare data for DeepParameters by discretizing continuous variables.
    
    Args:
        data (pd.DataFrame): Input dataset
        continuous_columns (list): List of continuous columns to discretize
        n_bins (int): Number of bins for discretization
    
    Returns:
        pd.DataFrame: Discretized dataset ready for DeepParameters
    """
    processed_data = data.copy()
    
    # Auto-detect continuous columns if not specified
    if continuous_columns is None:
        continuous_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Discretize continuous variables
    for col in continuous_columns:
        if col in processed_data.columns:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
            processed_data[col] = discretizer.fit_transform(processed_data[[col]]).astype(int)
    
    # Ensure categorical variables are properly encoded
    categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        processed_data[col] = le.fit_transform(processed_data[col])
    
    # Convert all columns to integer type (required for DeepParameters)
    for col in processed_data.columns:
        processed_data[col] = processed_data[col].astype(int)
    
    return processed_data

# Example usage
# Load your raw data
raw_data = pd.read_csv('your_raw_data.csv')

# Preprocess data
discretized_data = preprocess_data_for_deepparameters(raw_data, n_bins=3)

# Verify discretization
print("Data ready for DeepParameters!")
for col in discretized_data.columns:
    print(f"{col}: {sorted(discretized_data[col].unique())}")
```

## 📦 Installation

```bash
pip install deepparameters
# For the latest 2.0.5 features:
pip install --upgrade deepparameters
```

## 🎯 Quick Start

```python
from deepparameters import learn_cpd_for_node
import pandas as pd
# Import BayesianNetwork (DiscreteBayesianNetwork for newer pgmpy versions)
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

# Load and preprocess your data (ensure it's discretized!)
data = pd.read_csv('your_discretized_data.csv')

# Define your Bayesian network structures
true_model = BayesianNetwork([('A', 'B'), ('C', 'B')])
learnt_model = BayesianNetwork([('A', 'B'), ('C', 'B')])

# Learn CPD with default settings
cpd = learn_cpd_for_node(
    node='B', 
    data=data, 
    true_model=true_model, 
    learnt_bn_structure=learnt_model,
    num_parameters=10
)

# Advanced configuration with tunable optimizers and early stopping
cpd = learn_cpd_for_node(
    node='B',
    data=data,
    true_model=true_model,
    learnt_bn_structure=learnt_model,
    num_parameters=20,
    network_type='lstm',           # Try: simple, advanced, lstm, autoencoder, vae, bnn
    sampling_method='4',           # Try: 1-8 for different sampling methods
    optimizer='adamw',             # adam, adamw, sgd, rmsprop, nadam
    early_stopping_patience=15,   # Configurable early stopping
    epochs=200,
    verbose=True
)
```

## ⚡ Parallel Learning

Learn CPDs for entire networks using configurable parallel execution with two distinct approaches:

### **Topological Parallel Learning**
Groups nodes by dependency levels in the network. Nodes at the same topological level (same distance from root nodes) are learned in parallel. This approach:
- Works well for networks with clear hierarchical structure
- Suitable for data with strong dependency relationships
- Reliable for complex networks with multiple dependency paths

### **Parent-Child Factor Group Learning**
Groups nodes based on shared parent relationships. Nodes with the same parents are learned together. This approach:
- Optimized for networks with many nodes sharing common parents
- Suitable for data with clustered family relationships
- More efficient for hierarchical data structures

```python
from deepparameters.core import DeepParametersLearner

# Initialize learner
learner = DeepParametersLearner()

# Option 1: Topological Level Groups (default)
# Groups nodes by dependency levels - reliable for complex networks
cpds = learner.learn_network_parallel(
    data=data,
    network_structure=bn,
    parallel_style='topological',  # Default
    max_workers=4,
    verbose=True
)

# Option 2: Parent-Child Factor Groups (optimized)
# Groups nodes by parent relationships - better for hierarchical structures
cpds = learner.learn_network_parallel(
    data=data,
    network_structure=bn,
    parallel_style='parent_child',  # Optimized for hierarchical networks
    max_workers=4,
    verbose=True
)

# Advanced parallel configuration
cpds = learner.learn_network_parallel(
    data=data,
    network_structure=bn,
    parallel_style='parent_child',   # Choose decomposition strategy
    network_type='advanced',         # Neural architecture
    sampling_method='3',             # Importance sampling
    epochs=100,
    max_workers=6,                   # Parallel workers
    max_time_per_group=60,           # Time limit per group
    verbose=True
)
```

## 🏗️ Architecture Overview

### Neural Network Architectures

| Architecture | Description | Best For |
|-------------|-------------|----------|
| `simple` | Basic feedforward network | Quick prototyping |
| `advanced` | Multi-layer with dropout and batch norm | General purpose |
| `lstm` | Long Short-Term Memory network | Sequential dependencies |
| `autoencoder` | Encoder-decoder architecture | Feature learning |
| `vae` | Variational Autoencoder | Probabilistic modeling |
| `bnn` | Bayesian Neural Network | Uncertainty quantification |
| `normalizing_flow` | Normalizing Flow model | Complex distributions |
| `ultra` | Advanced hybrid architecture | High-performance scenarios |
| `mega` | Maximum complexity architecture | Research applications |

### Sampling Methods

| Method | ID | Description | Strengths |
|--------|-------|-------------|-----------|
| Gibbs | `1` | Gibbs sampling (MCMC chain) | Simple, reliable |
| Metropolis-Hastings | `2` | MCMC acceptance-rejection | Flexible |
| Importance | `3` | Weighted samples | Efficient for rare events |
| BPE | `4` | Belief Propagation Extension | Fast inference |
| Variational | `5` | Variational inference (optimization-based) | Scalable |
| HMC | `6` | Hamiltonian Monte Carlo (gradient-based) | High accuracy |
| SMC | `7` | Sequential Monte Carlo (particle filters) | Particle filtering |
| Adaptive KDE | `8` | Kernel Density Estimation (adaptive bandwidth) | Adaptive bandwidth |

## 📊 Performance Evaluation

DeepParameters provides comprehensive evaluation metrics:

- **Mean Absolute Error (MAE)**: Primary accuracy metric
- **KL Divergence**: Distribution similarity measure  
- **Root Mean Square Error (RMSE)**: Error magnitude
- **Maximum Error**: Worst-case performance
- **JS Divergence**: Symmetric distribution distance
- **Cosine Similarity**: Directional similarity
- **Probability Consistency**: Probabilistic validity

```python
from deepparameters import evaluate_cpd_performance

# Evaluate learned CPD against ground truth
results = evaluate_cpd_performance(learned_cpd, true_cpd)
print(f"MAE: {results['mean_absolute_error']:.4f}")
print(f"KL Divergence: {results['kl_divergence']:.4f}")
```

## 🔧 Advanced Configuration

```python
# Full parameter configuration with detailed comments
cpd = learn_cpd_for_node(
    node='B',                       # Target node to learn CPD for
    data=data,                      # Training dataset (must be discretized)
    true_model=true_model,          # True Bayesian network structure
    learnt_bn_structure=learnt_model, # Learned network structure
    num_parameters=50,              # Number of parameters for neural network
    network_type='vae',             # Neural architecture type
    sampling_method='8',            # Sampling method (1-8)
    optimizer='adamw',              # Optimizer: adam, adamw, sgd, rmsprop, nadam
    early_stopping_patience=20,    # Early stopping patience (epochs)
    epochs=500,                     # Maximum training epochs
    batch_size=64,                  # Training batch size
    learning_rate=0.001,            # Learning rate for optimizer
    validation_split=0.2,           # Validation data split ratio
    early_stopping=True,            # Enable early stopping
    verbose=True,                   # Enable verbose output
    random_state=42                 # Random seed for reproducibility
)
```

📋 BASIC USAGE
──────────────
```python
from deepparameters.core import DeepParametersLearner

learner = DeepParametersLearner()

# For hierarchical networks (recommended)
cpds = learner.learn_network_parallel(
    data=your_data,
    network_structure=your_network,
    parallel_style='parent_child'
)

# For complex interconnected networks (recommended)  
cpds = learner.learn_network_parallel(
    data=your_data,
    network_structure=your_network,
    parallel_style='topological'
)
```

🎯 STYLE SELECTION GUIDE
────────────────────────
┌─────────────────────┬─────────────────┬──────────────────┐
│ Your Network Type   │ Recommended     │ Why?             │
├─────────────────────┼─────────────────┼──────────────────┤
│ Family trees        │ parent_child    │ Natural hierarchy│
│ Organization charts │ parent_child    │ Clear parent-child│
│ Social networks     │ topological     │ Complex cross-deps│
│ Knowledge graphs    │ topological     │ Intricate patterns│
│ Unknown structure   │ topological     │ Safe default     │
└─────────────────────┴─────────────────┴──────────────────┘

⚙️ ADVANCED CONFIGURATION
─────────────────────────
```python
# High-performance configuration
cpds = learner.learn_network_parallel(
    data=data,
    network_structure=network,
    parallel_style='parent_child',
    max_workers=4,           # Optimal for most systems
    epochs=30,               # Good balance of quality/speed
    network_type='advanced', # For complex learning
    max_time_per_group=60   # Prevent timeouts
)

# Performance benchmarking
results = learner.benchmark_parallel_performance(
    data=data,
    network_structure=network,
    parallel_style='parent_child',
    max_workers_list=[1, 2, 4],
    epochs=20
)
```


🎯 BEST PRACTICES
─────────────────
1. Start with parallel_style='topological' if unsure
2. Use 2-4 workers for optimal performance
3. Provide 500+ samples for reliable learning
4. Monitor memory usage for large networks
5. Implement error handling in production code


### 🔧 Neural Network Optimizer Options

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `adam` | Adaptive moment estimation | General purpose (default) |
| `adamw` | Adam with weight decay | Better generalization |
| `sgd` | Stochastic gradient descent | Simple, reliable |
| `rmsprop` | Root mean square propagation | Recurrent networks |
| `nadam` | Nesterov-accelerated Adam | Faster convergence |

## 📚 Documentation

- **[Complete Documentation](https://github.com/rudzanimulaudzi/DeepParameters)**: Full documentation and API reference
- **[Performance Analysis](https://github.com/rudzanimulaudzi/DeepParameters/blob/main/PERFORMANCE_ANALYSIS_REPORT.md)**: Detailed benchmarks and comparisons
- **[API Reference](https://github.com/rudzanimulaudzi/DeepParameters/blob/main/API_REFERENCE.md)**: Complete function documentation

## 🧪 Example Workflows

**[Complete Workflow Guide](https://github.com/rudzanimulaudzi/DeepParameters/blob/main/DEEPPARAMETERS_WORKFLOW_GUIDE.md)**: Step-by-step usage examples and comprehensive tutorials

## 🤝 Contributing

We welcome contributions! For now email rudzani.mulaudzi2@students.wits.ac.za

## 📄 License

This project is licensed under the MIT License.

## 🎓 Citation

If you use DeepParameters in your research, please cite:

```bibtex
@software{deepparameters2025,
  title={DeepParameters: Neural Network Bayesian Network CPD Learning},
  author={Rudzani Mulaudzi},
  year={2025},
  version={2.0.5},
  url={https://github.com/rudzanimulaudzi/DeepParameters}
}
```

## 🆘 Support

Coming Soon

---

**DeepParameters** - Making advanced CPD learning accessible to everyone.
