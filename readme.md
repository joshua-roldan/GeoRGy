# GeoRGy: Spiking Neural Networks with Manifold Analysis on SHD Dataset

## Abstract

This project explores spiking neural networks (SNNs) using `snnTorch` for processing the Spiking Heidelberg Digits (SHD) dataset, with a focus on manifold analysis to understand the learned representations. Spiking neural networks are biologically-inspired models that process information through discrete spike events, offering energy-efficient alternatives to traditional artificial neural networks. The SHD dataset provides a challenging benchmark for temporal pattern recognition tasks using neuromorphic data.

The project implements SNN architectures using `snnTorch`, a PyTorch-based library for building and training spiking neural networks. Through manifold analysis techniques, we investigate the geometric structure of the learned feature representations, examining how the network organizes and separates different classes in the high-dimensional space. This analysis provides insights into the network's internal dynamics and can inform architecture design and training strategies.

Key components include:
- **Spiking Neural Network Implementation**: Using `snnTorch` to build and train SNNs on the SHD dataset
- **Manifold Analysis**: Applying dimensionality reduction and geometric analysis techniques to visualize and understand the learned representations
- **Temporal Pattern Recognition**: Processing neuromorphic spike trains to classify spoken digits

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install tonic matplotlib numpy
```

### Example Command Lines

#### Download and Visualize SHD Dataset

```bash
python download_and_visualize_shd.py
```

This will download the SHD dataset and create raster plots of the first three samples, saving them to the `./data/` folder.

#### Download and Prepare the SHD Dataset (Alternative)

```bash
python -m snntorch.datasets.shd --data_path ./data/shd
```

#### Train a Spiking Neural Network

```bash
python train_snn.py --dataset shd --epochs 50 --batch_size 64 --learning_rate 0.001
```

#### Run Manifold Analysis

```bash
python analyze_manifold.py --model_path ./models/snn_shd.pth --output_dir ./results/manifold
```

#### Visualize Results

```bash
python visualize_manifold.py --results_dir ./results/manifold --save_path ./figures/manifold_plot.png
```

#### Complete Pipeline Example

```bash
# Train the model
python train_snn.py --dataset shd --epochs 50 --batch_size 64

# Extract features and perform manifold analysis
python analyze_manifold.py --model_path ./models/snn_shd.pth

# Generate visualizations
python visualize_manifold.py --results_dir ./results/manifold
```

## Dataset

The Spiking Heidelberg Digits (SHD) dataset consists of spoken digits (0-9) encoded as spike trains, providing a neuromorphic benchmark for temporal pattern recognition tasks.

## References

- `snnTorch`: [https://github.com/jeshraghian/snntorch](https://github.com/jeshraghian/snntorch)
- SHD Dataset: [https://zenodo.org/record/1219636](https://zenodo.org/record/1219636)
