#!/usr/bin/env python
"""
Quick test script to verify the installation works correctly.
This creates random data and runs a simple manifold analysis.
"""

import numpy as np
from mftma.manifold_analysis_correlation import manifold_analysis_corr

print("=" * 60)
print("Testing Neural Manifolds Replica MFT Installation")
print("=" * 60)

# Create random test data: 100 classes, 50 examples each, 5000 features
print("\n1. Creating random test data...")
np.random.seed(0)
X = [np.random.randn(5000, 50) for i in range(100)]
print(f"   Created {len(X)} classes with {X[0].shape[1]} examples each")
print(f"   Each example has {X[0].shape[0]} features")

# Run the analysis
print("\n2. Running manifold analysis...")
print("   This may take a minute...")
kappa = 0  # No margin
n_t = 200  # Number of samples for analysis

capacity_all, radius_all, dimension_all, center_correlation, K = manifold_analysis_corr(X, kappa, n_t)

# Compute averages
avg_capacity = 1/np.mean(1/capacity_all)
avg_radius = np.mean(radius_all)
avg_dimension = np.mean(dimension_all)

# Display results
print("\n3. Results:")
print("=" * 60)
print(f"   Average Capacity (α_M):  {avg_capacity:.4f}")
print(f"   Average Radius (R_M):   {avg_radius:.4f}")
print(f"   Average Dimension (D_M): {avg_dimension:.4f}")
print(f"   Center Correlation:     {center_correlation:.4f}")
print(f"   Optimal K:              {K}")
print("=" * 60)

print("\n✅ Installation verified! The code is working correctly.")
print("\nNext steps:")
print("  - Read GETTING_STARTED.md for detailed instructions")
print("  - Try the example notebook: examples/MFTMA_VGG16_example.ipynb")
print("  - Analyze your own models using the provided templates")

