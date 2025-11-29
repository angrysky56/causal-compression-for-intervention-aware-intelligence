# Causal Compression for Intervention Aware Intelligence

## Overview
This project explores the formalization and implementation of **Intervention Aware Intelligence**, focusing on the problem of "Catastrophic Forgetting" in continuous learning systems. It proposes and validates a mechanism called **Projected Plasticity** (or the **Continuity Field**) to maintain a system's identity (causal invariance) while allowing it to adapt to new information.

The core innovation is a geometric approach where "Identity" is defined as a manifold of stable states, and "Plasticity" (learning) is constrained to the orthogonal complement of this manifold (the local tangent space).

## Key Concepts
- **Identity Manifold**: A geometric representation of the system's core "Self" or causal structure.
- **Projected Plasticity**: An update rule that projects learning gradients onto the orthogonal complement of the Identity Manifold, ensuring that adaptation does not erode the system's identity.
- **Continuity Field**: The software component that implements the Identity Manifold and calculates "Drift" (deviation from the manifold).

## Directory Structure
- `src/causal_compression/`: Core source code.
    - `continuity_field.py`: Implementation of the Continuity Field using a simulated Vector DB and Local Tangent Space Projection.
- `experiments/`: Experiment scripts.
    - `iron_creche.py`: The "Iron Creche" experiment demonstrating the mechanism on a toy LSTM model.
- `tests/`: Unit tests.
- `docs/`: Documentation and analysis files.
    - `TKUI_FORMALIZATION.md`: Formal theoretical framework.
    - `tkui_integration_analysis.md`: Mapping of theory to implementation.

## Installation
Requires Python 3.8+ and the following libraries:
```bash
pip install torch numpy scikit-learn
```

## Usage

### Running the Experiment
To run the "Iron Creche" experiment, which compares standard training vs. Projected Plasticity:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 experiments/iron_creche.py
```

### Running Tests
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m unittest discover tests
```

## License
[lol, you are going to steal it anyway]

Tyler B. Hall aka angrysky56

