# DeepParameters API Reference

## Table of Contents

1. [Core Functions](#core-functions)
2. [Neural Architectures](#neural-architectures)
3. [Sampling Methods](#sampling-methods)
4. [Utility Functions](#utility-functions)
5. [Data Structures](#data-structures)
6. [Error Handling](#error-handling)
7. [Configuration Parameters](#configuration-parameters)
8. [Examples](#examples)

## Core Functions

### learn_cpd_for_node

The primary function for learning Conditional Probability Distributions using neural networks.

```python
def learn_cpd_for_node(
    node: str,
    data: pd.DataFrame,
    true_model: BayesianNetwork,
    learnt_bn_structure: BayesianNetwork,
    num_parameters: int,
    network_type: str = 'simple',
    sampling_method: str = '1',
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.0,
    early_stopping: bool = False,
    verbose: bool = False,
    random_state: int = None
) -> TabularCPD
```

#### Parameters

**Required Parameters:**

- **node** (`str`): The target node for which to learn the CPD
  - Must exist in both true_model and learnt_bn_structure
  - Should be a valid node name (string identifier)

- **data** (`pd.DataFrame`): Training dataset
  - Must contain columns for the target node and all its parents
  - Values should be discrete integers (0, 1, 2, ...)
  - Minimum recommended: 100 samples per variable

- **true_model** (`BayesianNetwork`): The true underlying Bayesian network structure
  - Used for comparison and evaluation
  - Must be a valid pgmpy BayesianNetwork object

- **learnt_bn_structure** (`BayesianNetwork`): The learned or hypothesized network structure
  - Structure used for CPD learning
  - Can be same as true_model or different

- **num_parameters** (`int`): Number of parameters for the neural network
  - Recommended range: 10-100
  - Higher values increase model capacity but also computational cost

**Optional Parameters:**

- **network_type** (`str`, default: 'simple'): Neural network architecture
  - Options: 'simple', 'advanced', 'lstm', 'autoencoder', 'vae', 'bnn', 'normalizing_flow', 'ultra', 'mega'
  - See [Neural Architectures](#neural-architectures) for details

- **sampling_method** (`str`, default: '1'): Sampling strategy
  - Options: '1', '2', '3', '4', '5', '8'
  - See [Sampling Methods](#sampling-methods) for details

- **epochs** (`int`, default: 100): Number of training epochs
  - Range: 50-500 (depending on complexity)
  - More epochs may improve accuracy but increase training time

- **batch_size** (`int`, default: 32): Training batch size
  - Range: 16-128
  - Larger batches may speed up training but require more memory

- **learning_rate** (`float`, default: 0.001): Learning rate for optimization
  - Range: 0.0001-0.01
  - Lower values for more stable training, higher for faster convergence

- **validation_split** (`float`, default: 0.0): Fraction of data for validation
  - Range: 0.0-0.3
  - Used for monitoring overfitting and early stopping

- **early_stopping** (`bool`, default: False): Enable early stopping
  - Requires validation_split > 0.0
  - Stops training when validation performance stops improving

- **verbose** (`bool`, default: False): Enable verbose output
  - Shows training progress and diagnostic information
  - Useful for debugging and monitoring

- **random_state** (`int`, default: None): Random seed for reproducibility
  - Use same value for consistent results across runs

#### Returns

**TabularCPD**: Learned conditional probability distribution
- pgmpy TabularCPD object containing learned probabilities
- Can be used with pgmpy BayesianNetwork for inference
- Contains probability values normalized to sum to 1.0

#### Raises

- **ValueError**: Invalid parameters or data format
- **KeyError**: Missing required columns in data
- **RuntimeError**: Training failure or convergence issues
- **MemoryError**: Insufficient memory for large models

#### Example Usage

```python
from deepparameters import learn_cpd_for_node
import pandas as pd
from pgmpy.models import BayesianNetwork

# Prepare data
data = pd.DataFrame({
    'A': [0, 1, 0, 1, 0, 1],
    'B': [0, 0, 1, 1, 0, 1],
    'C': [0, 1, 1, 0, 1, 0]
})

# Define network
model = BayesianNetwork([('A', 'C'), ('B', 'C')])

# Learn CPD
cpd = learn_cpd_for_node(
    node='C',
    data=data,
    true_model=model,
    learnt_bn_structure=model,
    num_parameters=20,
    network_type='advanced',
    sampling_method='5',
    epochs=150,
    verbose=True
)

print(cpd)
```

## Neural Architectures

### Simple Neural Network

Basic feedforward network for quick prototyping.

```python
class SimpleNN:
    """
    Simple neural network with basic dense layers.
    
    Architecture:
    - Input layer
    - Hidden layer (num_parameters neurons)
    - Output layer with softmax activation
    
    Best for:
    - Quick prototyping
    - Baseline comparisons
    - Simple probability distributions
    
    Characteristics:
    - Fast training
    - Low memory usage
    - Stable convergence
    """
```

### Advanced Neural Network

Enhanced network with regularization and normalization.

```python
class AdvancedNN:
    """
    Advanced neural network with regularization techniques.
    
    Architecture:
    - Input layer
    - Hidden layers with dropout and batch normalization
    - Output layer with softmax activation
    
    Features:
    - Dropout for regularization
    - Batch normalization
    - L2 regularization
    
    Best for:
    - General-purpose applications
    - Balanced speed and accuracy
    - Most CPD learning tasks
    """
```

### LSTM Architecture

Long Short-Term Memory network for sequential dependencies.

```python
class LSTMNetwork:
    """
    LSTM-based architecture for sequential patterns.
    
    Architecture:
    - LSTM layers
    - Dense output layer
    - Temporal attention mechanisms
    
    Features:
    - Memory of previous states
    - Sequential pattern recognition
    - Temporal dependencies
    
    Best for:
    - Sequential data
    - Time-series CPDs
    - Temporal dependencies
    """
```

### Variational Autoencoder (VAE)

Probabilistic encoder-decoder architecture.

```python
class VAENetwork:
    """
    Variational Autoencoder for probabilistic modeling.
    
    Architecture:
    - Encoder network
    - Latent space with reparameterization
    - Decoder network
    
    Features:
    - Probabilistic latent representation
    - Uncertainty quantification
    - Generative capabilities
    
    Best for:
    - Complex probability distributions
    - Uncertainty modeling
    - High-accuracy requirements
    """
```

### Bayesian Neural Network (BNN)

Neural network with Bayesian inference.

```python
class BayesianNN:
    """
    Bayesian Neural Network with uncertainty quantification.
    
    Architecture:
    - Probabilistic weights
    - Variational inference
    - Monte Carlo sampling
    
    Features:
    - Built-in uncertainty
    - Robust to overfitting
    - Principled probabilistic approach
    
    Best for:
    - Uncertainty quantification
    - Small datasets
    - Robust predictions
    """
```

### Normalizing Flow

Invertible neural transformation.

```python
class NormalizingFlow:
    """
    Normalizing Flow for exact likelihood modeling.
    
    Architecture:
    - Invertible transformations
    - Coupling layers
    - Exact likelihood computation
    
    Features:
    - Exact likelihood
    - Invertible transformations
    - Complex distribution modeling
    
    Best for:
    - Exact likelihood requirements
    - Complex distributions
    - Research applications
    """
```

## Sampling Methods

### Method 1: Gibbs Sampling

```python
def gibbs_sampling(model, evidence, num_samples=1000):
    """
    Gibbs sampling for MCMC inference.
    
    Parameters:
    - model: Bayesian network model
    - evidence: Observed evidence
    - num_samples: Number of samples to generate
    
    Returns:
    - Samples from posterior distribution
    
    Characteristics:
    - Fast convergence
    - Simple implementation
    - Good for discrete variables
    """
```

### Method 2: Metropolis-Hastings

```python
def metropolis_hastings_sampling(model, evidence, num_samples=1000):
    """
    Metropolis-Hastings MCMC sampling.
    
    Parameters:
    - model: Bayesian network model
    - evidence: Observed evidence
    - num_samples: Number of samples to generate
    
    Returns:
    - MCMC samples
    
    Characteristics:
    - Flexible proposal mechanisms
    - General-purpose MCMC
    - Good theoretical properties
    """
```

### Method 3: Importance Sampling

```python
def importance_sampling(model, evidence, num_samples=1000):
    """
    Importance sampling for inference.
    
    Parameters:
    - model: Bayesian network model
    - evidence: Observed evidence
    - num_samples: Number of samples to generate
    
    Returns:
    - Weighted samples
    
    Characteristics:
    - Effective for rare events
    - No Markov chain
    - Parallel computation friendly
    """
```

### Method 4: BPE (Belief Propagation Extension)

```python
def bpe_sampling(model, evidence, num_samples=1000):
    """
    Belief Propagation Extension sampling.
    
    Parameters:
    - model: Bayesian network model
    - evidence: Observed evidence
    - num_samples: Number of samples to generate
    
    Returns:
    - Approximate samples
    
    Characteristics:
    - Fastest method
    - Exact for trees
    - Low computational overhead
    """
```

### Method 5: Variational Inference

```python
def variational_sampling(model, evidence, num_samples=1000):
    """
    Variational inference sampling.
    
    Parameters:
    - model: Bayesian network model
    - evidence: Observed evidence
    - num_samples: Number of samples to generate
    
    Returns:
    - Variational approximation samples
    
    Characteristics:
    - Deterministic optimization
    - Scalable to large problems
    - Good approximation quality
    """
```

### Method 8: Hamiltonian Monte Carlo (HMC)

```python
def hmc_sampling(model, evidence, num_samples=1000):
    """
    Hamiltonian Monte Carlo sampling.
    
    Parameters:
    - model: Bayesian network model
    - evidence: Observed evidence
    - num_samples: Number of samples to generate
    
    Returns:
    - High-quality HMC samples
    
    Characteristics:
    - Highest accuracy
    - Uses gradient information
    - Excellent convergence
    - More computationally expensive
    """
```

## Utility Functions

### evaluate_cpd_performance

```python
def evaluate_cpd_performance(learned_cpd, true_cpd):
    """
    Evaluate CPD learning performance.
    
    Parameters:
    - learned_cpd: TabularCPD learned by the model
    - true_cpd: Ground truth TabularCPD
    
    Returns:
    - dict: Performance metrics
    
    Metrics included:
    - mean_absolute_error
    - kl_divergence
    - rmse
    - max_error
    - js_divergence
    - cosine_similarity
    - probability_consistency
    """
```

### validate_input_data

```python
def validate_input_data(data, node, parents):
    """
    Validate input data format and consistency.
    
    Parameters:
    - data: Input DataFrame
    - node: Target node name
    - parents: List of parent node names
    
    Returns:
    - bool: True if valid, raises exception otherwise
    
    Checks:
    - Required columns present
    - Data types are appropriate
    - No missing values in critical columns
    - Value ranges are valid
    """
```

### format_cpd_output

```python
def format_cpd_output(probabilities, node, parents):
    """
    Format probability values into TabularCPD.
    
    Parameters:
    - probabilities: Numpy array of probabilities
    - node: Target node name
    - parents: List of parent nodes
    
    Returns:
    - TabularCPD: Formatted CPD object
    
    Features:
    - Automatic normalization
    - Proper variable ordering
    - Evidence card calculation
    """
```

## Data Structures

### TabularCPD Structure

```python
class TabularCPD:
    """
    Conditional Probability Distribution representation.
    
    Attributes:
    - variable: str - Target variable name
    - variable_card: int - Cardinality of target variable
    - values: np.array - Probability values
    - evidence: list - Parent variables
    - evidence_card: list - Cardinalities of parents
    
    Methods:
    - marginalize()
    - normalize()
    - get_value()
    """
```

### BayesianNetwork Structure

```python
class BayesianNetwork:
    """
    Bayesian Network representation.
    
    Attributes:
    - nodes: list - Network nodes
    - edges: list - Directed edges
    - cpds: list - Conditional probability distributions
    
    Methods:
    - add_node()
    - add_edge()
    - add_cpds()
    - check_model()
    """
```

## Error Handling

### Common Exceptions

#### ValueError
```python
# Raised when invalid parameters are provided
try:
    cpd = learn_cpd_for_node(
        node='InvalidNode',  # Node not in network
        data=data,
        true_model=model,
        learnt_bn_structure=model,
        num_parameters=10
    )
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

#### KeyError
```python
# Raised when required columns are missing from data
try:
    cpd = learn_cpd_for_node(
        node='C',
        data=incomplete_data,  # Missing required columns
        true_model=model,
        learnt_bn_structure=model,
        num_parameters=10
    )
except KeyError as e:
    print(f"Missing column: {e}")
```

#### RuntimeError
```python
# Raised when training fails or convergence issues occur
try:
    cpd = learn_cpd_for_node(
        node='C',
        data=data,
        true_model=model,
        learnt_bn_structure=model,
        num_parameters=1000,  # Too many parameters
        epochs=10000  # Too many epochs
    )
except RuntimeError as e:
    print(f"Training failed: {e}")
```

#### MemoryError
```python
# Raised when insufficient memory for large models
try:
    cpd = learn_cpd_for_node(
        node='C',
        data=huge_data,  # Very large dataset
        true_model=model,
        learnt_bn_structure=model,
        num_parameters=10000,  # Very large model
        batch_size=10000  # Very large batch
    )
except MemoryError as e:
    print(f"Insufficient memory: {e}")
```

## Configuration Parameters

### Architecture-Specific Parameters

#### Simple/Advanced Networks
```python
config = {
    'hidden_layers': [num_parameters],
    'activation': 'relu',
    'dropout_rate': 0.2,
    'l2_regularization': 0.01
}
```

#### LSTM Networks
```python
config = {
    'lstm_units': num_parameters,
    'return_sequences': False,
    'dropout': 0.2,
    'recurrent_dropout': 0.2
}
```

#### VAE Networks
```python
config = {
    'encoder_dims': [num_parameters, num_parameters//2],
    'latent_dim': num_parameters//4,
    'decoder_dims': [num_parameters//2, num_parameters],
    'beta': 1.0  # KL divergence weight
}
```

#### BNN Networks
```python
config = {
    'prior_std': 1.0,
    'posterior_mu_init': 0.0,
    'posterior_rho_init': -3.0,
    'kl_weight': 1.0
}
```

### Sampling-Specific Parameters

#### MCMC Methods (Gibbs, MH)
```python
config = {
    'burn_in': 100,
    'thinning': 1,
    'chain_length': 1000
}
```

#### HMC Sampling
```python
config = {
    'step_size': 0.01,
    'num_leapfrog_steps': 10,
    'target_accept_prob': 0.8
}
```

#### Variational Inference
```python
config = {
    'learning_rate': 0.001,
    'num_steps': 1000,
    'convergence_threshold': 1e-6
}
```

## Examples

### Basic Usage Example

```python
from deepparameters import learn_cpd_for_node
import pandas as pd
from pgmpy.models import BayesianNetwork

# Create sample data
data = pd.DataFrame({
    'A': [0, 1, 0, 1, 0, 1, 0, 1],
    'B': [0, 0, 1, 1, 0, 0, 1, 1],
    'C': [0, 1, 1, 0, 1, 0, 0, 1]
})

# Define network
model = BayesianNetwork([('A', 'C'), ('B', 'C')])

# Learn CPD with default settings
cpd = learn_cpd_for_node(
    node='C',
    data=data,
    true_model=model,
    learnt_bn_structure=model,
    num_parameters=10
)

print("Learned CPD:")
print(cpd)
```

### Advanced Configuration Example

```python
# Advanced usage with custom parameters
cpd_advanced = learn_cpd_for_node(
    node='C',
    data=data,
    true_model=model,
    learnt_bn_structure=model,
    num_parameters=50,
    network_type='vae',
    sampling_method='8',
    epochs=200,
    batch_size=64,
    learning_rate=0.0005,
    validation_split=0.2,
    early_stopping=True,
    verbose=True,
    random_state=42
)
```

### Error Handling Example

```python
try:
    cpd = learn_cpd_for_node(
        node='C',
        data=data,
        true_model=model,
        learnt_bn_structure=model,
        num_parameters=20,
        network_type='vae',
        sampling_method='8',
        epochs=150,
        verbose=True
    )
    
    print("CPD learned successfully")
    print(cpd)
    
except ValueError as e:
    print(f"Parameter error: {e}")
except KeyError as e:
    print(f"Data error: {e}")
except RuntimeError as e:
    print(f"Training error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Evaluation Example

```python
from deepparameters import evaluate_cpd_performance
from pgmpy.factors.discrete import TabularCPD

# Create ground truth CPD
true_cpd = TabularCPD(
    variable='C', variable_card=2,
    values=[[0.8, 0.4, 0.6, 0.2],
            [0.2, 0.6, 0.4, 0.8]],
    evidence=['A', 'B'],
    evidence_card=[2, 2]
)

# Learn CPD
learned_cpd = learn_cpd_for_node(
    node='C', data=data, true_model=model,
    learnt_bn_structure=model, num_parameters=20
)

# Evaluate performance
metrics = evaluate_cpd_performance(learned_cpd, true_cpd)

print("Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

This API reference provides comprehensive documentation for all functions, classes, and configuration options in DeepParameters. Use this reference to understand parameter meanings, expected behaviors, and implementation details for effective usage of the package.