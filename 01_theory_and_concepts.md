# DeepParameters: Theory and Concepts

**Written for humans — no PhD required.**

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [Bayesian Networks — A Gentle Introduction](#2-bayesian-networks--a-gentle-introduction)
3. [What Is a Conditional Probability Distribution (CPD)?](#3-what-is-a-conditional-probability-distribution-cpd)
4. [The Old Way: Counting and MLE](#4-the-old-way-counting-and-mle)
5. [The DeepParameters Way: Neural Networks Learn CPDs](#5-the-deepparameters-way-neural-networks-learn-cpds)
6. [What "Sampling" Actually Means Here](#6-what-sampling-actually-means-here)
7. [The 12 Sampling / Refinement Methods](#7-the-12-sampling--refinement-methods)
8. [Parallel Learning — Doing Many Things at Once](#8-parallel-learning--doing-many-things-at-once)
9. [Performance Metrics](#9-performance-metrics)
10. [What Makes DeepParameters Novel](#10-what-makes-deepparameters-novel)
11. [End-to-End Example with Real Numbers](#11-end-to-end-example-with-real-numbers)

---

## 1. The Problem We're Solving

Imagine you want to build a computer model that can answer questions like:

> "Given that a patient has a fever and a cough, what is the probability they have the flu?"

To answer that you need a **probabilistic model** — one that captures the relationships between symptoms and diseases. A Bayesian network is exactly that kind of model.

Building a Bayesian network requires two things:
1. **Structure** — a diagram showing which variables influence which others (edges in a graph).
2. **Parameters** — the actual probability numbers that quantify those relationships.

Learning the parameters — specifically the *conditional probability distributions* (CPDs) — is the hard part. Traditional methods work poorly when:
- You have sparse data (not many rows in your table)
- Relationships between variables are complex and non-linear
- The network has many variables and parents

**DeepParameters solves this by using neural networks to learn CPDs instead of simple counting.**

---

## 2. Bayesian Networks — A Gentle Introduction

A Bayesian Network (BN) is a directed graph where:
- Each **node** represents a variable (e.g., Fever, Cough, Disease)
- Each **edge** (arrow) represents a direct influence (e.g., Disease → Fever means disease causes fever)
- Each node has a **CPD** that says: *"Given the states of my parents, what are my own probabilities?"*

### Example: Medical Diagnosis Network

```
Fever ────┐
Cough ────┼──► Disease
Fatigue ──┘
```

This graph says that Fever, Cough, and Fatigue all influence Disease. The arrows point *into* Disease, meaning Disease is the "child" and the symptoms are its "parents" in probabilistic language.

**Key rule:** Bayesian networks must be DAGs — Directed Acyclic Graphs. No cycles allowed. Information only flows one way.

### Why Bayesian Networks are Useful

Once a BN is learned from data, you can:
- Ask "what is P(Disease=sick | Fever=yes, Cough=no)?" → **inference**
- Find the most probable explanation for observed symptoms → **diagnosis**
- Simulate what happens if you intervene (e.g., give medication) → **causal reasoning**

---

## 3. What Is a Conditional Probability Distribution (CPD)?

A CPD is just a table of probabilities. For every combination of its parent variables' states, it says: "given those parent states, here's the probability of my own states."

### Example CPD for "Disease"

| Fever | Cough | Fatigue | P(Disease=Sick) | P(Disease=Healthy) |
|-------|-------|---------|-----------------|---------------------|
| No    | No    | No      | 0.05            | 0.95               |
| No    | No    | Yes     | 0.20            | 0.80               |
| No    | Yes   | No      | 0.15            | 0.85               |
| No    | Yes   | Yes     | 0.40            | 0.60               |
| Yes   | No    | No      | 0.30            | 0.70               |
| Yes   | No    | Yes     | 0.55            | 0.45               |
| Yes   | Yes   | No      | 0.60            | 0.40               |
| Yes   | Yes   | Yes     | 0.90            | 0.10               |

Each row must sum to 1.0 (you're always either sick or healthy — just with different probabilities).

### How Many Entries Does a CPD Have?

For a node with:
- **k** possible states
- Parents with cardinalities p₁, p₂, ..., pₙ

Total CPD entries = k × p₁ × p₂ × ... × pₙ

For Disease (2 states) with 3 binary parents: 2 × 2 × 2 × 2 = **16 numbers** to learn.

---

## 4. The Old Way: Counting and MLE

The classical approach is Maximum Likelihood Estimation (MLE). You count how often each combination appears in your data:

```
P(Disease=Sick | Fever=Yes, Cough=Yes, Fatigue=Yes) 
  = count(Sick, Fever, Cough, Fatigue) / count(Fever, Cough, Fatigue)
```

**Problems with pure counting:**

1. **Sparse data problem:** If you only have 500 rows but 8 parent combinations, some combinations might appear 0 times. You'd get 0/0 — undefined probability.

2. **No generalisation:** Each cell of the CPD is estimated independently. The network can't "learn" that having two symptoms is worse than one. It just counts.

3. **Scaling problem:** With many parents (10 binary parents = 1024 combinations), you'd need thousands of rows per combination to get a reliable estimate.

**Bayesian Parameter Learning** adds a "prior" (a starting assumption) to avoid zeros, but still doesn't generalise well.

---

## 5. The DeepParameters Way: Neural Networks Learn CPDs

DeepParameters replaces counting with a **neural network** for each node.

### The Key Idea

Instead of: "Count how many times this specific combination appeared"

We do: "Train a neural network to predict P(node | parents) given any combination"

The neural network **generalises** across parent combinations. It learns patterns like "more symptoms = higher probability of disease" rather than treating each combination independently.

### The Pipeline

```
Raw Data
    │
    ▼
[Data Validation & Discretisation]
    │
    ▼
[Neural Network Training]
    │  (learns the relationship between parents and child)
    ▼
[CPD Construction via Systematic Sampling]
    │  (query the neural network for every parent combination)
    ▼
[CPD Refinement via Statistical Methods]
    │  (Gibbs, MH, Importance sampling, etc.)
    ▼
TabularCPD — ready to use in pgmpy
```

### Why Discrete Data?

DeepParameters requires **discrete (categorical) data**. All variable values must be integers representing categories (0, 1, 2...).

This is because CPDs are discrete probability tables — each row corresponds to a specific discrete state. If your data is continuous, you must **discretise** it first (e.g., convert height measurements into bins: "short", "medium", "tall" → 0, 1, 2).

---

## 6. What "Sampling" Actually Means Here

This is one of the most misunderstood parts. The word "sampling" in DeepParameters does **not** mean sampling random rows from your dataset.

**It means systematically querying the trained neural network to build the CPD table.**

Think of the trained neural network as an "oracle". Once trained, you ask it:

> "What's P(Disease | Fever=0, Cough=0, Fatigue=0)?"  
> "What's P(Disease | Fever=0, Cough=0, Fatigue=1)?"  
> "What's P(Disease | Fever=1, Cough=1, Fatigue=1)?"  
> ... (all 8 combinations)

You collect all the answers and arrange them into the CPD table. This is the "sampling" step — sampling the neural network's knowledge.

After this initial CPD is built, a **refinement step** (the statistical sampling methods) is optionally applied to smooth and improve it.

---

## 7. The 12 Sampling / Refinement Methods

After the neural network produces an initial CPD, one of 12 refinement methods is applied. You select one using the `sampling_method` parameter (as a string number).

### Method 1: Gibbs Sampling (`sampling_method='1'`)

**What it is:** A Markov Chain Monte Carlo (MCMC) method.

**How it works:** It runs a chain of random samples from the whole Bayesian network. At each step, it updates one variable at a time while keeping all others fixed. After many steps, the sampled values follow the true joint distribution.

**When to use:** When you want the CPD to be fully consistent with the rest of the network's joint distribution.

**Analogy:** Imagine a committee where each member changes their mind one at a time, always trying to agree with everyone else. After enough rounds, consensus reflects the group's true beliefs.

---

### Method 2: Metropolis-Hastings (`sampling_method='2'`)

**What it is:** Another MCMC method with an acceptance-rejection step.

**How it works:** 
1. Propose a small random change to the current CPD parameters.
2. If the new version fits the data better → always accept.
3. If it's worse → accept with some probability (proportional to how much worse).
4. Repeat thousands of times; average the accepted samples.

**When to use:** Good general-purpose refiner. Explores the probability space broadly.

**Analogy:** A hiker trying to reach the highest mountain peak. They mostly go uphill but occasionally accept a downhill step, which helps them escape local valleys.

---

### Method 3: Importance Sampling (`sampling_method='3'`)

**What it is:** A weighted averaging technique.

**How it works:** 
1. Draw many random valid CPDs from a simple "proposal" distribution.
2. Weight each random CPD by how well it fits the actual data.
3. Take the weighted average as the final CPD.

**When to use:** When you want a fast, parallelisable refinement.

**Analogy:** Asking 1000 random people for their opinion, then weighting the opinions of more knowledgeable people more heavily.

---

### Method 4: Bayesian Parameter Estimation (`sampling_method='4'`)

**What it is:** The classical Bayesian approach using Dirichlet priors.

**How it works:** 
1. Count how many times each (parent_state, node_state) combination appears in the data.
2. Add a small constant (the Dirichlet prior α) to every count to avoid zeros.
3. Normalise to get probabilities.

**When to use:** When you want a simple, interpretable, data-driven estimate without MCMC complexity. Works very well when combined with neural network output as a starting point.

**Formula:**  
$P(\text{node} = k \mid \text{parents} = j) = \frac{N_{jk} + \alpha}{\sum_k (N_{jk} + \alpha)}$ where $N_{jk}$ = count of observations.

---

### Method 5: Variational Inference (`sampling_method='5'`)

**What it is:** An optimisation-based alternative to MCMC.

**How it works:** Instead of sampling, it finds the best approximating distribution by minimising the difference (KL divergence) between a simple family of distributions and the true posterior. Uses gradient descent-like updates.

**When to use:** Faster than MCMC for large networks; trades some accuracy for speed.

**Analogy:** Instead of exploring every path through a forest, you draw the straightest possible road through it.

---

### Method 6: Hamiltonian Monte Carlo / HMC (`sampling_method='6'`)

**What it is:** A sophisticated MCMC method using gradient information.

**How it works:** Adds "momentum" to the Metropolis-Hastings random walk. The gradients of the log-likelihood guide the proposals, making them much more efficient (fewer rejected steps, better exploration).

**When to use:** When MCMC accuracy matters and you can afford the computational cost of gradient calculations.

**Analogy:** Instead of random stumbling, the hiker uses a compass and physics to follow the mountain's slope directly.

---

### Method 7: Sequential Monte Carlo / SMC (`sampling_method='7'`)

**What it is:** A particle filter approach.

**How it works:** Maintains a population of "particles" (possible CPDs). Processes the training data sequentially — each data point updates the particle weights. Periodically "resamples" to keep good particles and discard bad ones.

**When to use:** When data arrives in streams or you want a method that naturally handles sequential updates.

---

### Method 8: Adaptive KDE (`sampling_method='8'`)

**What it is:** Kernel Density Estimation applied to CPD refinement.

**How it works:** Fits a smooth continuous density function over discrete data counts using Gaussian kernels. The bandwidth (smoothing width) is chosen automatically via cross-validation.

**When to use:** When data is sparse for some parent combinations and you want smooth interpolation.

---

### Method 9: Weighted Sampling (`sampling_method='9'`)

**What it is:** The simplest refinement method.

**How it works:** Treats the neural network output directly as likelihoods, normalises them column-by-column, and applies minor smoothing to remove numerical noise.

**When to use:** As a fast baseline or when the neural network output is already high quality.

---

### Method 10: Stratified Sampling (`sampling_method='10'`)

**What it is:** Divides parameter space into strata and samples proportionally.

**How it works:** Sorts the neural network parameter distribution into equal-sized bands (strata). Samples from each band in proportion to its size. Prevents a few dominant parameters from overwhelming the estimate.

**When to use:** When CPD parameters span a wide range of magnitudes.

---

### Method 11: KDE Sampling (`sampling_method='11'`)

**What it is:** Similar to Adaptive KDE but uses cross-validation more aggressively to find the optimal bandwidth.

**How it works:** Fits a KDE over collected node-state probabilities, samples from it, and maps samples back into the CPD structure.

---

### Method 12: Dirichlet Bayesian Sampling (`sampling_method='12'`)

**What it is:** A Bayesian method using the Dirichlet distribution as a conjugate prior to the multinomial.

**How it works:** 
1. For each parent configuration, computes Dirichlet concentration parameters by combining the prior α, observed counts from data, and pseudo-counts from the neural network output.
2. Draws many samples from this Dirichlet distribution.
3. Averages the samples to get the final probabilities.

**When to use:** Best for proper Bayesian uncertainty quantification. The neural network output is incorporated as informative pseudo-counts, making this the most theoretically principled method.

**Formula:**  
$\theta_{jk} \sim \text{Dirichlet}(\alpha_0 + N_{jk} + \text{NN\_pseudocounts}_{jk})$

---

## 8. Parallel Learning — Doing Many Things at Once

A Bayesian network usually has many nodes. Learning CPDs one-by-one is slow. DeepParameters can learn CPDs for multiple nodes at the same time (in parallel).

### Why Parallelism Is Valid

Nodes in a Bayesian network are **conditionally independent** given their parents. This means that once you know a node's parent values, learning its CPD doesn't depend on learning any other node's CPD simultaneously.

The network's DAG structure tells us which nodes can be safely learned at the same time.

### Two Parallel Styles

**Style 1: Topological (`parallel_style='topological'`)**

Groups nodes by their "depth" in the graph. All nodes at depth 0 (no parents, root nodes) are learned together in the first batch. Then all nodes at depth 1 (whose parents are root nodes) are learned together. And so on.

```
Batch 1: [RootA, RootB, RootC]       ← all learned in parallel
Batch 2: [ChildOfA, ChildOfB]        ← learned in parallel after batch 1
Batch 3: [GrandchildOfA]             ← learned after batch 2
```

**Best for:** Complex interconnected networks where some nodes have parents at different levels.

**Style 2: Parent-Child (`parallel_style='parent_child'`)**

Groups nodes by their parent-child relationship clusters. A parent and its children form one group; independent branches are separate groups.

**Best for:** Hierarchical or tree-like networks. Often 20-40% faster than topological for these structures.

### Time-Bounded Execution

You can set `max_time_per_group` (in seconds) to prevent any single node from consuming too much time. If a complex node exceeds the time limit, learning stops early and returns what was learned so far.

---

## 9. Performance Metrics

After learning, DeepParameters evaluates the quality of the learned CPD against the true CPD using 7 metrics:

| Metric | What It Measures | Perfect Value |
|--------|------------------|---------------|
| **MAE** | Mean Absolute Error — average absolute difference between learned and true probabilities | 0.0 |
| **RMSE** | Root Mean Squared Error — penalises large errors more than MAE | 0.0 |
| **KL Divergence** | How much the learned distribution differs from the true one (information-theoretic) | 0.0 |
| **JS Divergence** | A symmetrised, smoothed version of KL — bounded between 0 and 1 | 0.0 |
| **Cosine Similarity** | How parallel the probability vectors are | 1.0 |
| **Max Error** | Worst single probability estimate | 0.0 |
| **Probability Consistency** | Whether learned probabilities sum to 1 per column | True |

**KL Divergence** is the gold standard for comparing probability distributions:

$$KL(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Where P is the true CPD and Q is the learned CPD. KL = 0 means perfect match; higher values mean more difference.

---

## 10. What Makes DeepParameters Novel

DeepParameters combines several techniques in a way that, as of the time of writing, has not been done systematically before:

### Innovation 1: Neural Networks → Complete CPD Tables

Previous work used neural networks for classification (single predictions). DeepParameters is designed to systematically **query a trained neural network for every possible parent combination** and assemble the results into a complete, standard `TabularCPD` that any pgmpy-based tool can use.

### Innovation 2: 9 Architectures × 12 Sampling Methods = 108 Combinations

DeepParameters allows systematic experimentation across a matrix of neural architectures and statistical refinement methods. No prior tool exposed this level of configurability in a unified API.

### Innovation 3: Production-Ready Pipeline

The package goes from raw data → trained model → industry-standard CPD object → ready to deploy. Previous academic work produced research prototypes, not installable Python packages with a clean API.

### Innovation 4: Parallel Learning with Factor Group Decomposition

Decomposing network structure into independently learnable groups and running them in parallel, with time-bounded execution, is novel in the BN parameter-learning context.

---

## 11. End-to-End Example with Real Numbers

Let's trace through a complete example.

### Setup

```python
import pandas as pd
import numpy as np
from deepparameters import learn_cpd_for_node

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

# Network structure: Fever and Cough → Disease
true_model = BayesianNetwork([('Fever', 'Disease'), ('Cough', 'Disease')])
learnt_model = BayesianNetwork([('Fever', 'Disease'), ('Cough', 'Disease')])

# Dataset: 500 patients with discrete values (0 or 1)
data = pd.DataFrame({
    'Fever':   [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, ...],   # 500 rows
    'Cough':   [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, ...],
    'Disease': [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, ...]
})
```

### Step 1: Call the Main Function

```python
cpd = learn_cpd_for_node(
    node='Disease',
    data=data,
    true_model=true_model,
    learnt_bn_structure=learnt_model,
    num_parameters=10,
    network_type='simple',       # Use Simple NN
    sampling_method='4',         # Bayesian Parameter Estimation
    epochs=100,
    verbose=False
)
```

### Step 2: What Happens Internally

1. **Data prep:** The code identifies that 'Disease' has parents ['Fever', 'Cough'].
   - `X = data[['Fever', 'Cough']].values`  → shape (500, 2)
   - `y = data['Disease'].values`           → shape (500,)

2. **Network build (SimpleCPDLearner):** An MLP classifier with layers [64, 32] is constructed. It takes 2 inputs (Fever, Cough values) and outputs P(Disease=0) and P(Disease=1).

3. **Training:** Standard gradient descent for 100 epochs.

4. **CPD construction:** Generate all parent combinations:
   - [0, 0] → network predicts [0.94, 0.06] → P(Disease | Fever=0, Cough=0) = [0.94, 0.06]
   - [0, 1] → network predicts [0.77, 0.23]
   - [1, 0] → network predicts [0.68, 0.32]
   - [1, 1] → network predicts [0.15, 0.85]

5. **Refinement (BPE):** The Bayesian Parameter Estimator counts actual occurrences in data, adds α=1 prior, and blends with the neural network output.

6. **Output:** A `TabularCPD` object:

```
P(Disease | Fever, Cough)

         Fever=0,Cough=0   Fever=0,Cough=1   Fever=1,Cough=0   Fever=1,Cough=1
Disease=0     0.9312            0.7605            0.6801            0.1423
Disease=1     0.0688            0.2395            0.3199            0.8577
```

Each column sums to 1.0. ✓

### Step 3: Use in pgmpy

```python
from pgmpy.inference import VariableElimination

true_model.add_cpds(cpd)
infer = VariableElimination(true_model)
result = infer.query(variables=['Disease'], evidence={'Fever': 1, 'Cough': 1})
print(result)  # P(Disease=1 | Fever=1, Cough=1) ≈ 0.86
```

---

## Summary

DeepParameters is a Python package that replaces traditional counting-based CPD learning in Bayesian networks with neural network-based learning + statistical refinement. It:

- Takes discrete data as input
- Trains one neural network per node (from 9 architecture options)
- Systematically queries the network to build a full CPD table
- Refines the CPD using one of 12 statistical methods
- Returns a standard `TabularCPD` object compatible with pgmpy
- Can learn all nodes in parallel for speed

It is designed to work better than pure MLE when data is sparse, relationships are non-linear, or when you need to experiment systematically across many architecture/sampling combinations.
