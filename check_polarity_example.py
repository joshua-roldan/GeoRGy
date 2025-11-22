#!/usr/bin/env python
"""
Example script to check polarity values in different neuromorphic datasets.
This demonstrates why some datasets have multiple polarities and others don't.
"""

import numpy as np
import tonic

print("=" * 70)
print("Polarity Analysis in Neuromorphic Datasets")
print("=" * 70)

# 1. Check SHD (Audio Dataset) - Should have only polarity = 1
print("\n1. Spiking Heidelberg Dataset (SHD) - Audio Dataset")
print("-" * 70)
try:
    shd_dataset = tonic.datasets.SHD(save_to='./data', train=True)
    sample_events, label = shd_dataset[0]
    
    # Check polarity values
    if hasattr(sample_events, 'dtype') and sample_events.dtype.names:
        # Structured array
        if 'p' in sample_events.dtype.names:
            polarities = sample_events['p']
        elif 'polarity' in sample_events.dtype.names:
            polarities = sample_events['polarity']
        else:
            print("   Could not find polarity field in structured array")
            print(f"   Available fields: {sample_events.dtype.names}")
            polarities = None
    else:
        # Regular array - assume format is (t, x, y, p) or (t, neuron, p)
        if sample_events.shape[1] >= 3:
            polarities = sample_events[:, -1]  # Last column is usually polarity
        else:
            polarities = None
    
    if polarities is not None:
        unique_polarities = np.unique(polarities)
        print(f"   Sample label: {label}")
        print(f"   Number of events: {len(sample_events)}")
        print(f"   Unique polarity values: {unique_polarities}")
        print(f"   Polarity distribution:")
        for p in unique_polarities:
            count = np.sum(polarities == p)
            percentage = 100 * count / len(polarities)
            print(f"     Polarity {p}: {count} events ({percentage:.1f}%)")
        print("\n   → SHD is an AUDIO dataset, so polarity is always 1")
        print("   → Audio signals don't need ON/OFF distinction")
except Exception as e:
    print(f"   Error loading SHD: {e}")

# 2. Try to check a vision dataset (if available)
print("\n2. Vision Datasets (DVS) - Would Have Multiple Polarities")
print("-" * 70)
print("   Vision datasets like N-MNIST, DVS Gesture typically have:")
print("   - Polarity = 1: ON events (brightness increases)")
print("   - Polarity = 0: OFF events (brightness decreases)")
print("\n   Example vision datasets in Tonic:")
print("   - tonic.datasets.NMNIST()")
print("   - tonic.datasets.DVSGesture()")
print("   - tonic.datasets.NCaltech101()")
print("\n   These would show both polarity 0 and 1 in their events")

# 3. Explain the difference
print("\n3. Why the Difference?")
print("-" * 70)
print("   VISION (DVS sensors):")
print("   - Need to detect both brightness INCREASES and DECREASES")
print("   - ON events: object getting brighter, light turning on")
print("   - OFF events: object getting darker, shadows appearing")
print("   - Both are essential for understanding motion and edges")
print("\n   AUDIO (like SHD):")
print("   - Audio waveforms oscillate around zero")
print("   - Important info is in MAGNITUDE and TIMING, not direction")
print("   - Single polarity is sufficient for spike representation")
print("   - The neuron/channel index encodes frequency information")

# 4. How to handle in your analysis
print("\n4. Implications for Manifold Analysis")
print("-" * 70)
print("   For SHD (single polarity):")
print("   - You can ignore the polarity field (it's always 1)")
print("   - Focus on time and neuron/channel dimensions")
print("\n   For vision datasets (multiple polarities):")
print("   - Option 1: Treat polarity as an additional feature dimension")
print("   - Option 2: Separate ON and OFF events into different manifolds")
print("   - Option 3: Create separate analyses for each polarity type")

print("\n" + "=" * 70)
print("See POLARITY_EXPLANATION.md for more details")

