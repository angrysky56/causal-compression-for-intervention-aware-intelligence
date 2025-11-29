# Causal Compression for Intervention-Aware Intelligence

**A Unified Framework for Robust, Interpretable, and Aligned AGI**

> "The minimum description length of intelligence is not $H(X)$ but $H(X | \text{do}(\cdot))$—the entropy of causally sufficient information."

## Overview

This repository houses the research and implementation of **Compression Intelligence**, a paradigm that unifies abstractive compression with causal reasoning. It addresses the fundamental fragility of statistical intelligence (e.g., LLMs) by architecturally enforcing **Causal-Statistical Alignment**.

Current AI systems optimize for correlation ($I(X; Y)$), making them vulnerable to spurious correlations and distribution shifts. This project implements systems that optimize for intervention ($I(X; Y | \text{do}(\cdot))$), ensuring that compressed representations retain the causal structure of the world.

## Core Concepts

### 1. Causal-Statistical Alignment Index (CSAI)
A formal metric measuring how well a compressed representation preserves causal information under intervention.
$$ \text{CSAI} = \frac{I(f_\theta(X); Y | \text{do}(X_j))}{I(f_\theta(X); Y)} $$

### 2. The Orthogonality Thesis
We prove that statistical compression (Shannon entropy minimization) and causal understanding are orthogonal optimization objectives. Achieving AGI requires a unified architecture that explicitly bridges this gap.

### 3. Iron Creche Architecture
The reference implementation of this framework. It uses a **Teacher-Student** setup with **Contrastive Causal Triplets** to train models that distinguish between:
- **Causal Pivots**: Variables that change outcomes under intervention.
- **Nuisance Variables**: Spurious correlations or stylistic noise.

### 4. Continuity Field (Identity Manifold)
A mechanism to resolve the "Ship of Theseus" paradox in learning systems. It maintains a persistent definition of "Self" (Identity Manifold) while allowing for structural adaptation (Plasticity) via orthogonal projection.

## Repository Structure

```
.
├── src/
│   └── causal_compression/     # Core library for causal compression primitives
│       ├── continuity_field.py # Identity Manifold implementation
│       └── ...
├── experiments/
│   └── iron_creche.py          # Reference implementation of the Iron Creche architecture
├── tests/                      # Unit tests for core components
├── docs/                       # Theoretical papers and formalizations
│   ├── compression_intelligence.md # The foundational paper
│   └── TKUI_FORMALIZATION.md       # Formalization of the Triadic Kernel
└── pyproject.toml              # Project configuration and dependencies
```

## Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv pip install -r pyproject.toml
# OR if using standard pip
pip install .
```

## Usage

### Running the Iron Creche Experiment
The `iron_creche.py` script demonstrates the core training loop, including the **Continuity Field** for identity preservation.

```bash
python experiments/iron_creche.py
```

### Using the Continuity Field
The `ContinuityField` class can be used in any PyTorch training loop to track and constrain manifold drift.

```python
from causal_compression.continuity_field import ContinuityField

# Initialize
cf = ContinuityField(embedding_dim=64, k_neighbors=10)

# Add anchor states (stable identity)
cf.add_anchor(initial_state_vector)

# Check drift during training
drift = cf.get_drift_metric(current_state_vector)
print(f"Manifold Drift: {drift}")
```

## License

[License Information]

**Tyler B. Hall** (angrysky56)
