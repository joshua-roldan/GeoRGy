#!/usr/bin/env python
"""
Download Spiking Heidelberg Dataset using Tonic and perform manifold analysis
on two batches of the training set.
"""

import numpy as np
import tonic
import tonic.transforms as transforms
import ssl
from mftma.manifold_analysis_correlation import manifold_analysis_corr

print("=" * 70)
print("Spiking Heidelberg Dataset - Manifold Analysis")
print("=" * 70)

# Step 1: Download and load the SHD dataset with ToFrame transform
print("\n1. Downloading Spiking Heidelberg Dataset...")
print("   This may take a few minutes on first run...")

# Handle SSL certificate issues (common on macOS)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

save_directory = './data'

# Get sensor size for SHD dataset
sensor_size = tonic.datasets.SHD.sensor_size
print(f"   Sensor size: {sensor_size}")

# Define ToFrame transform to convert events to time-binned frames
# n_time_bins determines the number of time bins
n_time_bins = 1  # Adjust this to control temporal resolution
frame_transform = transforms.ToFrame(
    sensor_size=sensor_size,
    n_time_bins=n_time_bins
)

# First, download and save the SHD dataset WITHOUT any transform, so we can access raw events for plotting
try:
    train_dataset_events = tonic.datasets.SHD(
        save_to=save_directory,
        train=True,
        transform=None
    )
except Exception as e:
    print(f"\n   Error downloading dataset (events): {e}")
    print("   If you see SSL certificate errors, you may need to:")
    print("   1. Install certificates: /Applications/Python\\ 3.12/Install\\ Certificates.command")
    print("   2. Or manually download the dataset from:")
    print("      https://zenodo.org/record/1219637")
    raise

print(f"   Raw event dataset loaded! Total samples: {len(train_dataset_events)}")

# As example, select the events from the first sample for plotting later
sample_events, sample_label = train_dataset_events[0]  # (events, label)
print(f"   Example sample events extracted for plotting (label={sample_label})")

# Step 1.5: Compute the number of spikes for each neuron over an entire audio and store it in a vector
print("\n1.5. Computing spike counts per neuron for all samples...")

num_of_unit_spikes_for_samples = []
n_neurons = sensor_size[0]  # Number of neurons/channels (e.g., 700 for SHD)


## Make function for filtering data based on labels.
def filter_by_labels(dataset, labels_to_include=None, max_samples_per_class=None, max_total_samples=None):
    """
    Filter dataset samples based on their labels, and sort the returned data by ascending label.
    
    Args:
        dataset: Tonic dataset object
        labels_to_include: List of labels to include, or None for all labels.
                          Can also be a range like range(10) for labels 0-9
        max_samples_per_class: Maximum number of samples per class (None = no limit)
        max_total_samples: Maximum total number of samples to return (None = no limit)
    
    Returns:
        filtered_samples: List of filtered samples (events or frames), sorted by label
        filtered_labels: List of corresponding labels (sorted ascending)
        label_counts: Dictionary with count of samples per label
    """
    filtered_pairs = []
    label_counts = {}
    
    # Convert labels_to_include to a set for fast lookup
    if labels_to_include is not None:
        if isinstance(labels_to_include, range):
            labels_to_include = set(labels_to_include)
        else:
            labels_to_include = set(labels_to_include)
    
    for idx in range(len(dataset)):
        sample, label = dataset[idx]
        
        # Check if label should be included
        if labels_to_include is not None and label not in labels_to_include:
            continue
        
        # Check if we've reached max samples per class
        if max_samples_per_class is not None:
            if label_counts.get(label, 0) >= max_samples_per_class:
                continue
        
        # Add sample
        filtered_pairs.append((sample, label))
        label_counts[label] = label_counts.get(label, 0) + 1
        
        # Check if we've reached max total samples
        if max_total_samples is not None and len(filtered_pairs) >= max_total_samples:
            break

    # Sort by label ascending
    filtered_pairs.sort(key=lambda x: x[1])
    filtered_samples = [sample for sample, label in filtered_pairs]
    filtered_labels = [label for sample, label in filtered_pairs]

    return filtered_samples, filtered_labels, label_counts

# Example usage: Filter first 100 samples with labels 0-9, max 10 per class, sorted by label
print("\n   Filtering dataset...")
filtered_samples, filtered_labels, label_counts = filter_by_labels(
    train_dataset_events,
    labels_to_include=range(10),  # Only classes 0-9
    max_samples_per_class=10,     # Max 10 samples per class
    max_total_samples=100         # But stop at 100 total samples
)

print(f"   Filtered dataset: {len(filtered_samples)} samples")
print(f"   Label distribution: {label_counts}")

# Use filtered data for spike counting
print("\n   Computing spike counts for filtered samples...")


filtered_class_0label, filtered_class_0samples, filtered_class_0label_counts = filtered_labels[0:10], filtered_samples[0:10], label_counts
filtered_class_1label, filtered_class_1samples, filtered_class_1label_counts = filtered_labels[10:20], filtered_samples[10:20], label_counts
filtered_class_2label, filtered_class_2samples, filtered_class_2label_counts = filtered_labels[20:30], filtered_samples[20:30], label_counts
filtered_class_3label, filtered_class_3samples, filtered_class_3label_counts = filtered_labels[30:40], filtered_samples[30:40], label_counts
filtered_class_4label, filtered_class_4samples, filtered_class_4label_counts = filtered_labels[40:50], filtered_samples[40:50], label_counts
filtered_class_5label, filtered_class_5samples, filtered_class_5label_counts = filtered_labels[50:60], filtered_samples[50:60], label_counts
filtered_class_6label, filtered_class_6samples, filtered_class_6label_counts = filtered_labels[60:70], filtered_samples[60:70], label_counts
filtered_class_7label, filtered_class_7samples, filtered_class_7label_counts = filtered_labels[70:80], filtered_samples[70:80], label_counts
filtered_class_8label, filtered_class_8samples, filtered_class_8label_counts = filtered_labels[80:90], filtered_samples[80:90], label_counts
filtered_class_9label, filtered_class_9samples, filtered_class_9label_counts = filtered_labels[90:100], filtered_samples[90:100], label_counts


def compute_spike_counts(filtered_samples, filtered_labels, n_neurons):
    """
    Computes spike counts for each sample in filtered_samples.

    Args:
        filtered_samples (list): List of event lists per sample.
        filtered_labels (list): List of labels corresponding to the samples.
        n_neurons (int): Number of neurons (length of spike count vector).

    Returns:
        list: List of spike count vectors (numpy arrays), one per sample.
    """
    num_of_unit_spikes_for_samples = []
    for sample_id, (events, label) in enumerate(zip(filtered_samples, filtered_labels)):
        # Initialize spike count vector for this sample
        num_of_unit_spikes = np.zeros(n_neurons)

        # Count spikes per neuron
        if len(events) > 0:
            for e in events:
                neuron_idx = int(e['x'])
                if 0 <= neuron_idx < n_neurons:
                    num_of_unit_spikes[neuron_idx] += 1

        num_of_unit_spikes_for_samples.append(num_of_unit_spikes)
    return num_of_unit_spikes_for_samples

# Example usage:
num_of_unit_spikes_for_samples_class_0 = compute_spike_counts(filtered_class_0samples,filtered_class_0label, sensor_size[0])
num_of_unit_spikes_for_samples_class_1 = compute_spike_counts(filtered_class_1samples,filtered_class_1label, sensor_size[0])
num_of_unit_spikes_for_samples_class_2 = compute_spike_counts(filtered_class_2samples,filtered_class_2label, sensor_size[0])
num_of_unit_spikes_for_samples_class_3 = compute_spike_counts(filtered_class_3samples,filtered_class_3label, sensor_size[0])
num_of_unit_spikes_for_samples_class_4 = compute_spike_counts(filtered_class_4samples,filtered_class_4label, sensor_size[0])
num_of_unit_spikes_for_samples_class_5 = compute_spike_counts(filtered_class_5samples,filtered_class_5label, sensor_size[0])
num_of_unit_spikes_for_samples_class_6 = compute_spike_counts(filtered_class_6samples,filtered_class_6label, sensor_size[0])
num_of_unit_spikes_for_samples_class_7 = compute_spike_counts(filtered_class_7samples,filtered_class_7label, sensor_size[0])
num_of_unit_spikes_for_samples_class_8 = compute_spike_counts(filtered_class_8samples,filtered_class_8label, sensor_size[0])
num_of_unit_spikes_for_samples_class_9 = compute_spike_counts(filtered_class_9samples,filtered_class_9label, sensor_size[0])


num_of_unit_spikes_for_samples_class_0_T =np.transpose(num_of_unit_spikes_for_samples_class_0)
num_of_unit_spikes_for_samples_class_1_T =np.transpose(num_of_unit_spikes_for_samples_class_1)
num_of_unit_spikes_for_samples_class_2_T =np.transpose(num_of_unit_spikes_for_samples_class_2)
num_of_unit_spikes_for_samples_class_3_T =np.transpose(num_of_unit_spikes_for_samples_class_3)
num_of_unit_spikes_for_samples_class_4_T =np.transpose(num_of_unit_spikes_for_samples_class_4)
num_of_unit_spikes_for_samples_class_5_T =np.transpose(num_of_unit_spikes_for_samples_class_5)
num_of_unit_spikes_for_samples_class_6_T =np.transpose(num_of_unit_spikes_for_samples_class_6)
num_of_unit_spikes_for_samples_class_7_T =np.transpose(num_of_unit_spikes_for_samples_class_7)
num_of_unit_spikes_for_samples_class_8_T =np.transpose(num_of_unit_spikes_for_samples_class_8)
num_of_unit_spikes_for_samples_class_9_T =np.transpose(num_of_unit_spikes_for_samples_class_9)

X = [num_of_unit_spikes_for_samples_class_0_T, num_of_unit_spikes_for_samples_class_1_T, num_of_unit_spikes_for_samples_class_2_T, num_of_unit_spikes_for_samples_class_3_T, num_of_unit_spikes_for_samples_class_4_T, num_of_unit_spikes_for_samples_class_5_T, num_of_unit_spikes_for_samples_class_6_T, num_of_unit_spikes_for_samples_class_7_T, num_of_unit_spikes_for_samples_class_8_T, num_of_unit_spikes_for_samples_class_9_T]

# Step 5: Run manifold analysis on each batch
print("\n5. Running manifold analysis...")
print("   This may take a few minutes...")

kappa = 0  # No margin
n_t = 200  # Number of samples for analysis

print("\n   Analyzing Batch 1...")
capacity_all, radius_all, dimension_all, correlation_all, K_all = manifold_analysis_corr(
    X, kappa, n_t, n_reps=1
)


# Step 6: Compute and display results
print("\n6. Results:")
print("=" * 70)

# Batch 1 results
avg_capacity_all = 1/np.mean(1/capacity_all)
avg_radius_all = np.mean(radius_all)
avg_dimension_all = np.mean(dimension_all)

print("\nBatch 1 Results:")
print(f"   Average Capacity (α_M):  {avg_capacity_all:.4f}")
print(f"   Average Radius (R_M):   {avg_radius_all:.4f}")
print(f"   Average Dimension (D_M): {avg_dimension_all:.4f}")
print(f"   Center Correlation:     {correlation_all:.4f}")
print(f"   Optimal K:              {K_all}")


print("\n" + "=" * 70)
print("Analysis complete!")

