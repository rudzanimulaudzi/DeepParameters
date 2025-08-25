# DeepParameters Complete Workflow Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Basic Usage](#basic-usage)
4. [Advanced Configuration](#advanced-configuration)
5. [Neural Network Architectures](#neural-network-architectures)
6. [Sampling Methods](#sampling-methods)
7. [Performance Evaluation](#performance-evaluation)
8. [Complete Examples](#complete-examples)
9. [Troubleshooting](#troubleshooting)

## Introduction

DeepParameters is a comprehensive Python package for learning Conditional Probability Distributions (CPDs) in Bayesian networks using neural network architectures. This guide provides step-by-step instructions for using all features of the package.

## Installation and Setup

### Basic Installation

```python
pip install deepparameters
```

### Verify Installation

```python
import deepparameters
print(f"DeepParameters version: {deepparameters.__version__}")
```

### Required Dependencies

The package automatically installs:
- TensorFlow >= 2.8.0
- TensorFlow Probability >= 0.15.0
- pgmpy >= 0.1.19
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0

## Basic Usage

### Step 1: Import Required Libraries

```python
from deepparameters import learn_cpd_for_node
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
```

### Step 2: Prepare Your Data

```python
# Example: Create sample data
np.random.seed(42)
n_samples = 1000

# Generate synthetic data for Bayesian network A -> B <- C
data = pd.DataFrame({
    'A': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'C': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

# Generate B based on A and C
data['B'] = np.where(
    (data['A'] == 1) & (data['C'] == 1), 
    np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
    np.where(
        (data['A'] == 1) | (data['C'] == 1),
        np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    )
)

print("Data shape:", data.shape)
print("Data head:")
print(data.head())
```

### Step 3: Define Bayesian Network Structure

```python
# Define the true model structure
true_model = BayesianNetwork([('A', 'B'), ('C', 'B')])

# Define the learnt structure (can be same or different)
learnt_model = BayesianNetwork([('A', 'B'), ('C', 'B')])

print("True model edges:", true_model.edges())
print("Learnt model edges:", learnt_model.edges())
```

### Step 4: Learn CPD with Default Settings

```python
# Learn CPD for node B using default settings
cpd = learn_cpd_for_node(
    node='B',
    data=data,
    true_model=true_model,
    learnt_bn_structure=learnt_model,
    num_parameters=10
)

print("Learned CPD for node B:")
print(cpd)
```

## Advanced Configuration

### Specifying Neural Network Architecture

```python
# Use LSTM architecture
cpd_lstm = learn_cpd_for_node(
    node='B',
    data=data,
    true_model=true_model,
    learnt_bn_structure=learnt_model,
    num_parameters=20,
    network_type='lstm',
    epochs=100,
    verbose=True
)
```

### Specifying Sampling Method

```python
# Use Hamiltonian Monte Carlo sampling
cpd_hmc = learn_cpd_for_node(
    node='B',
    data=data,
    true_model=true_model,
    learnt_bn_structure=learnt_model,
    num_parameters=15,
    network_type='advanced',
    sampling_method='8',  # HMC
    epochs=150
)
```

### Complete Parameter Configuration

```python
cpd_full = learn_cpd_for_node(
    node='B',
    data=data,
    true_model=true_model,
    learnt_bn_structure=learnt_model,
    num_parameters=50,
    network_type='vae',
    sampling_method='5',  # Variational
    epochs=200,
    batch_size=64,
    learning_rate=0.001,
    validation_split=0.2,
    early_stopping=True,
    verbose=True,
    random_state=42
)
```

## Neural Network Architectures

### Available Architectures

| Architecture | Code | Best For | Complexity |
|-------------|------|----------|------------|
| Simple NN | 'simple' | Quick prototyping | Low |
| Advanced NN | 'advanced' | General purpose | Medium |
| LSTM | 'lstm' | Sequential data | Medium |
| Autoencoder | 'autoencoder' | Feature learning | Medium |
| VAE | 'vae' | Probabilistic modeling | High |
| BNN | 'bnn' | Uncertainty quantification | High |
| Normalizing Flow | 'normalizing_flow' | Complex distributions | High |
| Ultra | 'ultra' | High performance | Very High |
| Mega | 'mega' | Research applications | Very High |

### Architecture-Specific Examples

#### Simple Neural Network
```python
cpd_simple = learn_cpd_for_node(
    node='B', data=data, true_model=true_model, 
    learnt_bn_structure=learnt_model, num_parameters=10,
    network_type='simple', epochs=50
)
```

#### LSTM for Sequential Dependencies
```python
cpd_lstm = learn_cpd_for_node(
    node='B', data=data, true_model=true_model,
    learnt_bn_structure=learnt_model, num_parameters=20,
    network_type='lstm', epochs=100, batch_size=32
)
```

#### Bayesian Neural Network for Uncertainty
```python
cpd_bnn = learn_cpd_for_node(
    node='B', data=data, true_model=true_model,
    learnt_bn_structure=learnt_model, num_parameters=30,
    network_type='bnn', epochs=150, learning_rate=0.0005
)
```

## Sampling Methods

### Available Sampling Methods

| Method ID | Name | Description | Computational Cost |
|-----------|------|-------------|-------------------|
| '1' | Gibbs | Gibbs sampling | Low |
| '2' | Metropolis-Hastings | MCMC sampling | Medium |
| '3' | Importance | Importance sampling | Medium |
| '4' | BPE | Belief Propagation Extension | Low |
| '5' | Variational | Variational inference | Medium |
| '8' | HMC | Hamiltonian Monte Carlo | High |

### Sampling Method Examples

#### Gibbs Sampling (Fast, Reliable)
```python
cpd_gibbs = learn_cpd_for_node(
    node='B', data=data, true_model=true_model,
    learnt_bn_structure=learnt_model, num_parameters=15,
    sampling_method='1'
)
```

#### Hamiltonian Monte Carlo (High Accuracy)
```python
cpd_hmc = learn_cpd_for_node(
    node='B', data=data, true_model=true_model,
    learnt_bn_structure=learnt_model, num_parameters=25,
    sampling_method='8', epochs=200
)
```

## Performance Evaluation

### Built-in Evaluation Metrics

The package automatically computes several performance metrics:

- Mean Absolute Error (MAE)
- KL Divergence
- Root Mean Square Error (RMSE)
- Maximum Error
- JS Divergence
- Cosine Similarity
- Probability Consistency

### Custom Evaluation

```python
from deepparameters import evaluate_cpd_performance

# Create ground truth CPD for comparison
true_cpd = TabularCPD(
    variable='B', variable_card=2,
    values=[[0.8, 0.4, 0.6, 0.2],
            [0.2, 0.6, 0.4, 0.8]],
    evidence=['A', 'C'],
    evidence_card=[2, 2]
)

# Evaluate learned CPD
results = evaluate_cpd_performance(cpd, true_cpd)
print("Evaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

## Complete Examples

### Example 1: Medical Diagnosis Network

```python
import pandas as pd
import numpy as np
from deepparameters import learn_cpd_for_node
from pgmpy.models import BayesianNetwork

# Create medical diagnosis data
np.random.seed(42)
n_patients = 2000

# Generate symptoms and disease data
data = pd.DataFrame({
    'Fever': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
    'Cough': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
    'Fatigue': np.random.choice([0, 1], n_patients, p=[0.8, 0.2])
})

# Generate disease based on symptoms
disease_prob = (
    data['Fever'] * 0.4 + 
    data['Cough'] * 0.3 + 
    data['Fatigue'] * 0.2 + 
    np.random.normal(0, 0.1, n_patients)
)
data['Disease'] = (disease_prob > 0.3).astype(int)

# Define network structure
medical_network = BayesianNetwork([
    ('Fever', 'Disease'),
    ('Cough', 'Disease'), 
    ('Fatigue', 'Disease')
])

# Learn CPD for Disease using VAE
disease_cpd = learn_cpd_for_node(
    node='Disease',
    data=data,
    true_model=medical_network,
    learnt_bn_structure=medical_network,
    num_parameters=40,
    network_type='vae',
    sampling_method='5',
    epochs=150,
    verbose=True
)

print("Medical Diagnosis CPD:")
print(disease_cpd)
```

### Example 2: Financial Risk Assessment

```python
# Create financial data
financial_data = pd.DataFrame({
    'CreditScore': np.random.choice([0, 1, 2], 1500, p=[0.3, 0.5, 0.2]),
    'Income': np.random.choice([0, 1, 2], 1500, p=[0.4, 0.4, 0.2]),
    'DebtRatio': np.random.choice([0, 1], 1500, p=[0.6, 0.4])
})

# Generate risk based on financial factors
risk_score = (
    financial_data['CreditScore'] * 0.4 + 
    financial_data['Income'] * (-0.3) + 
    financial_data['DebtRatio'] * 0.5
)
financial_data['Risk'] = (risk_score > 1.0).astype(int)

# Define financial network
financial_network = BayesianNetwork([
    ('CreditScore', 'Risk'),
    ('Income', 'Risk'),
    ('DebtRatio', 'Risk')
])

# Learn CPD using BNN for uncertainty quantification
risk_cpd = learn_cpd_for_node(
    node='Risk',
    data=financial_data,
    true_model=financial_network,
    learnt_bn_structure=financial_network,
    num_parameters=35,
    network_type='bnn',
    sampling_method='8',
    epochs=200,
    learning_rate=0.0005
)

print("Financial Risk CPD:")
print(risk_cpd)
```

### Example 3: Multi-Architecture Comparison

```python
# Compare different architectures on the same problem
architectures = ['simple', 'advanced', 'lstm', 'vae', 'bnn']
results = {}

for arch in architectures:
    print(f"Training {arch} architecture...")
    
    cpd = learn_cpd_for_node(
        node='B',
        data=data,
        true_model=true_model,
        learnt_bn_structure=learnt_model,
        num_parameters=20,
        network_type=arch,
        epochs=100,
        verbose=False
    )
    
    results[arch] = cpd
    print(f"{arch} completed successfully")

# Display results
for arch, cpd in results.items():
    print(f"\n{arch.upper()} Architecture Result:")
    print(cpd)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Memory Errors with Large Datasets
```python
# Solution: Use smaller batch sizes and fewer parameters
cpd = learn_cpd_for_node(
    node='B', data=large_data, true_model=model, 
    learnt_bn_structure=model, num_parameters=10,
    batch_size=16, epochs=50
)
```

#### Issue 2: Convergence Issues
```python
# Solution: Adjust learning rate and use early stopping
cpd = learn_cpd_for_node(
    node='B', data=data, true_model=model,
    learnt_bn_structure=model, num_parameters=20,
    learning_rate=0.0001, early_stopping=True,
    validation_split=0.3, epochs=300
)
```

#### Issue 3: TensorFlow Graph Errors
```python
# Solution: Use simpler architectures for complex problems
cpd = learn_cpd_for_node(
    node='B', data=data, true_model=model,
    learnt_bn_structure=model, num_parameters=15,
    network_type='advanced',  # Instead of 'vae'
    sampling_method='1'       # Instead of complex sampling
)
```

### Performance Tips

1. **Start Simple**: Begin with 'simple' or 'advanced' architectures
2. **Tune Gradually**: Increase complexity only if needed
3. **Monitor Training**: Use verbose=True to track progress
4. **Use Validation**: Set validation_split for better generalization
5. **Experiment with Sampling**: Different methods work better for different problems

### Data Requirements

- **Minimum samples**: At least 100 samples per variable
- **Data types**: Discrete variables (0, 1, 2, etc.)
- **Missing values**: Handle before passing to the function
- **Balanced classes**: Consider class balance for better results

## Conclusion

This workflow guide covers the essential aspects of using DeepParameters. The package provides flexibility to experiment with different neural architectures and sampling methods while maintaining a simple, unified interface. Start with basic configurations and gradually explore advanced features as needed for your specific use case.