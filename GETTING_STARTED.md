# Getting Started with Neural Manifolds Replica MFT

## What This Code Does (Simple Explanation)

This codebase analyzes how neural networks organize information by studying **"manifolds"** - think of these as the shapes that different classes of data form in the network's internal representations.

### The Big Picture

Imagine you're looking at how a neural network "sees" different objects. For example:
- All images of "cats" form one shape (manifold) in the network's internal space
- All images of "dogs" form another shape
- All images of "cars" form yet another shape

This tool measures three key properties of these shapes:

1. **Capacity (α_M)**: How many different classes can the network separate? Higher capacity = can distinguish more classes
2. **Radius (R_M)**: How "spread out" is each class? Larger radius = more variation within a class
3. **Dimension (D_M)**: How many dimensions does each class actually use? Lower dimension = more efficient representation

### How It Works (Step by Step)

#### Step 1: Prepare Your Data
- You need a dataset with multiple classes (at least 30+ classes works best)
- For each class, you sample multiple examples (e.g., 50 images per class)
- The code extracts how the network represents each example at different layers

#### Step 2: Extract Activations
- The code runs your data through a neural network
- It captures the "activations" (internal representations) at each layer
- These activations show how the network transforms the input data

#### Step 3: Analyze Each Layer
For each layer, the code:
1. **Centers the data**: Removes the global mean to focus on differences
2. **Finds class centers**: Computes the average representation for each class
3. **Removes correlations**: Identifies and removes shared structure between classes
4. **Measures geometry**: Uses mathematical techniques (mean-field theory) to compute:
   - How separable the classes are (capacity)
   - How spread out each class is (radius)
   - How many dimensions each class uses (dimension)

#### Step 4: Interpret Results
- **Capacity increases** as you go deeper → network gets better at separating classes
- **Radius decreases** → classes become more compact/refined
- **Dimension decreases** → network compresses information more efficiently

## Installation

### Prerequisites
- Python 3.7-3.11 (Python 3.12+ requires updated dependencies - see below)
- pip package manager

**Note**: If you're using Python 3.12 or newer, the requirements have been updated to use compatible package versions. The original requirements used very old versions that don't work with Python 3.12+.

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `numpy` - for numerical computations
- `autograd` - for automatic differentiation (version 1.8.0+ for Python 3.12+)
- `scipy` - for scientific computing
- `pymanopt` - for optimization on manifolds
- `cvxopt` - for convex optimization
- `scikit-learn` - for machine learning utilities
- `torch` and `torchvision` - for PyTorch models (if using the example)

**Note for Python 3.12+ users**: After installing pymanopt, you may need to patch it for compatibility with newer scipy versions. The patch changes `scipy.misc.comb` to `scipy.special.comb` in the pymanopt rotations module. If you encounter import errors, see the troubleshooting section.

### Step 2: Install the Package

```bash
pip install -e .
```

The `-e` flag installs in "editable" mode, so changes to the code are immediately available.

## Quick Start Guide

### Option 1: Quick Test with Random Data

Create a simple Python script to test the installation:

```python
import numpy as np
from mftma.manifold_analysis_correlation import manifold_analysis_corr

# Create random data: 100 classes, 50 examples each, 5000 features
np.random.seed(0)
X = [np.random.randn(5000, 50) for i in range(100)]

# Run the analysis
kappa = 0  # No margin
n_t = 200  # Number of samples for analysis

capacity_all, radius_all, dimension_all, center_correlation, K = manifold_analysis_corr(X, kappa, n_t)

# Compute averages
avg_capacity = 1/np.mean(1/capacity_all)
avg_radius = np.mean(radius_all)
avg_dimension = np.mean(dimension_all)

print(f"Average Capacity: {avg_capacity:.4f}")
print(f"Average Radius: {avg_radius:.4f}")
print(f"Average Dimension: {avg_dimension:.4f}")
print(f"Center Correlation: {center_correlation:.4f}")
```

Save this as `test_quick.py` and run:
```bash
python test_quick.py
```

### Option 2: Analyze a PyTorch Model (Full Example)

The easiest way to get started is to use the example notebook:

1. **Open the example notebook**:
   ```bash
   jupyter notebook examples/MFTMA_VGG16_example.ipynb
   ```

2. **Or run it step by step**:
   - The notebook trains a VGG16 model on CIFAR-100
   - Extracts activations from each layer
   - Runs the manifold analysis
   - Plots the results

### Option 3: Analyze Your Own Model

Here's a template for analyzing your own PyTorch model:

```python
import torch
import numpy as np
from mftma.utils.analyze_pytorch import analyze
from mftma.manifold_analysis_correlation import manifold_analysis_corr

# 1. Load your model and dataset
model = your_model()  # Your PyTorch model
model.eval()  # Set to evaluation mode
dataset = your_dataset()  # Your PyTorch dataset

# 2. Run the analysis
results = analyze(
    model=model,
    dataset=dataset,
    sampled_classes=50,      # Number of classes to analyze
    examples_per_class=50,   # Examples per class
    kappa=0,                 # Margin (0 = no margin)
    n_t=300,                 # Number of samples for analysis
    layer_types=['Conv2d', 'Linear'],  # Which layer types to analyze
    projection=True,         # Project to lower dimension if needed
    projection_dimension=5000,
    seed=0
)

# 3. Extract and plot results
capacities = []
radii = []
dimensions = []

for layer_name, result in results.items():
    # Compute averages
    capacity = 1/np.mean(1/result['capacity'])
    radius = np.mean(result['radius'])
    dimension = np.mean(result['dimension'])
    
    capacities.append(capacity)
    radii.append(radius)
    dimensions.append(dimension)
    
    print(f"{layer_name}: Capacity={capacity:.4f}, Radius={radius:.4f}, Dimension={dimension:.4f}")

# 4. Plot the results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(capacities)
axes[0].set_ylabel('Capacity')
axes[0].set_title('Manifold Capacity by Layer')

axes[1].plot(radii)
axes[1].set_ylabel('Radius')
axes[1].set_title('Manifold Radius by Layer')

axes[2].plot(dimensions)
axes[2].set_ylabel('Dimension')
axes[2].set_title('Manifold Dimension by Layer')

plt.tight_layout()
plt.show()
```

## Understanding the Output

### Capacity (α_M)
- **What it means**: How many classes can be separated
- **What to look for**: 
  - Low capacity (< 0.1) = classes are hard to separate
  - High capacity (> 0.1) = classes are well separated
  - Usually increases as you go deeper in the network

### Radius (R_M)
- **What it means**: How spread out each class is
- **What to look for**:
  - Large radius = high variation within class
  - Small radius = compact, consistent representations
  - Usually decreases as you go deeper (classes become more refined)

### Dimension (D_M)
- **What it means**: How many dimensions each class actually uses
- **What to look for**:
  - High dimension = class uses many dimensions
  - Low dimension = class is compressed into fewer dimensions
  - Usually decreases as you go deeper (more efficient representation)

### Center Correlation (ρ_center)
- **What it means**: How similar the centers of different classes are
- **What to look for**:
  - High correlation = classes are similar
  - Low correlation = classes are distinct

## Data Format Requirements

Your data should be formatted as:
- A **list** of numpy arrays
- Each array has shape `(N, M)` where:
  - `N` = number of features (e.g., 5000)
  - `M` = number of examples per class (e.g., 50)
- One array per class
- **Important**: `M` should be less than `N` (fewer examples than features)

Example:
```python
X = [
    np.array([[feature1, feature2, ...],  # Example 1
              [feature1, feature2, ...],  # Example 2
              ...]),                       # Shape: (N, M1)
    np.array([[feature1, feature2, ...],  # Class 2, Example 1
              ...]),                       # Shape: (N, M2)
    # ... more classes
]
```

## Troubleshooting

### Common Issues

1. **"M should be less than N" error**
   - Solution: Use fewer examples per class, or project to lower dimensions

2. **Analysis takes too long**
   - Solution: Reduce `n_t` (e.g., from 300 to 200), or use projection to reduce dimensions

3. **Out of memory errors**
   - Solution: Use `projection=True` and set `projection_dimension=5000` or lower

4. **Dependencies not found**
   - Solution: Make sure you ran `pip install -r requirements.txt` and `pip install -e .`

## Next Steps

1. **Try the example notebook** to see the full workflow
2. **Experiment with different models** and datasets
3. **Compare different layers** to see how representations change
4. **Read the papers** referenced in the README for deeper understanding

## References

- Main paper: [Untangling in Invariant Speech Recognition (NeurIPS 2019)](https://arxiv.org/abs/2003.01787)
- Theory papers:
  - Classification and Geometry of General Perceptual Manifolds (Phys. Rev. X 2018)
  - Separability and Geometry of Object Manifolds in Deep Neural Networks (Nature Communications 2020)

## Getting Help

- Email: cory.stephenson@intel.com or sueyeonchung@gmail.com
- Check the example notebook for detailed usage examples

