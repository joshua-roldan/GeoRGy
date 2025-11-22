"""
Download SHD dataset using tonic and visualize raster plots of the first three samples.
"""

import matplotlib.pyplot as plt
import numpy as np
from tonic.datasets import SHD
import os
import ssl
import urllib.request

# Fix SSL certificate verification issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context


def download_shd_dataset(data_path="./data/shd"):
    """
    Download the SHD dataset using tonic.
    
    Args:
        data_path: Path where the dataset will be stored
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    print(f"Downloading SHD dataset to {data_path}...")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Download and load the SHD dataset
    train_dataset = SHD(save_to=data_path, train=True)
    test_dataset = SHD(save_to=data_path, train=False)
    
    print(f"Dataset downloaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def plot_raster(events, label, ax, title=None):
    """
    Plot a raster plot of spike events.
    
    Args:
        events: Array of events with shape (N, 4) where columns are [t, x, p, label]
                or structured array with fields ['t', 'x', 'p']
        label: Class label for the sample
        ax: Matplotlib axis to plot on
        title: Optional title for the plot
    """
    # Handle different event formats
    if isinstance(events, np.ndarray):
        if events.dtype.names is not None:
            # Structured array
            times = events['t']
            channels = events['x']
        else:
            # Regular array - assume format [t, x, p, ...]
            times = events[:, 0]
            channels = events[:, 1]
    else:
        # Try to convert to numpy array
        events = np.array(events)
        times = events[:, 0]
        channels = events[:, 1]
    
    # Plot raster
    ax.scatter(times, channels, s=0.5, c='black', alpha=0.6)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Channel', fontsize=10)
    
    if title is None:
        title = f'Label: {label}'
    else:
        title = f'{title} (Label: {label})'
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable limits
    if len(times) > 0:
        ax.set_xlim([times.min() - 10, times.max() + 10])
        ax.set_ylim([channels.min() - 1, channels.max() + 1])


def visualize_first_three_samples(dataset, save_path=None):
    """
    Visualize raster plots of the first three samples from the dataset.
    
    Args:
        dataset: Tonic dataset object
        save_path: Optional path to save the figure
    """
    print("\nVisualizing first three samples...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Raster Plots of First Three SHD Samples', fontsize=14, fontweight='bold')
    
    # Get first three samples
    for i in range(min(3, len(dataset))):
        events, label = dataset[i]
        
        print(f"Sample {i+1}: Label = {label}, Number of events = {len(events)}")
        
        # Plot raster
        plot_raster(events, label, axes[i], title=f'Sample {i+1}')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to download dataset and create visualizations."""
    # Set data path
    data_path = "./data/shd"
    
    # Download the dataset
    train_dataset, test_dataset = download_shd_dataset(data_path)
    
    # Visualize first three training samples
    print("\n" + "="*50)
    print("Training Set Samples:")
    print("="*50)
    visualize_first_three_samples(
        train_dataset, 
        save_path="./data/shd_train_samples.png"
    )
    
    # Optionally visualize first three test samples
    print("\n" + "="*50)
    print("Test Set Samples:")
    print("="*50)
    visualize_first_three_samples(
        test_dataset, 
        save_path="./data/shd_test_samples.png"
    )
    
    print("\n" + "="*50)
    print("Visualization complete!")
    print("="*50)


if __name__ == "__main__":
    main()

