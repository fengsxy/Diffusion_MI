# DMI - Diffusion-based Mutual Information Estimation

A PyTorch Lightning implementation of Mutual Information Neural Diffusion (MMG) Estimator with both unified and Hierarchical Mixture of Experts (MOE) architectures for robust mutual information estimation.

## Overview

This project implements novel approaches to mutual information estimation using diffusion models. It provides two distinct architectures:

1. **Unified Model** (`MMG_Unet.py`): Single denoising network handling all noise levels
2. **Mixture of Experts (MOE)** (`MMG_Unet_MOE.py`): Separate specialized models for different Signal-to-Noise Ratio (SNR) regions

### Key Features

- **Dual Architecture Support**: Choose between unified model or MOE approach
- **Diffusion-based Estimation**: Leverages diffusion models for robust MI estimation
- **Multiple Task Support**: Supports multinormal, half-cube, and spiral tasks
- **Exponential Moving Average (EMA)**: Optional EMA for training stability
- **Flexible Configuration**: Extensive hyperparameter customization
- **Automatic Experiment Management**: Built-in result tracking and checkpoint handling

## Architecture Comparison

### Unified Model (`MMG_Unet.py`)
- Single denoising network processes all samples
- Simpler architecture with fewer parameters
- Faster training and inference
- Good baseline performance

### MOE Model (`MMG_Unet_MOE.py`)
- Separate models for low SNR (< threshold) and high SNR (≥ threshold) samples
- Specialized expertise for different noise conditions
- More parameters but potentially better performance
- Automatic SNR-based routing

```python
# MOE routing logic
if low_mask.any():
    eps_hat_x_low = self.model_low(z_low, logsnr_low)
    eps_hat_y_low = self.model_low(z_low, logsnr_low, y_low)
    
if high_mask.any():
    eps_hat_x_high = self.model_high(z_high, logsnr_high)
    eps_hat_y_high = self.model_high(z_high, logsnr_high, y_high)
```

## Quick Start

### 1. Environment Setup

**Create and activate Conda environment:**
```bash
conda create -n dmi_env python=3.8
conda activate dmi_env
```

**Install dependencies:**
```bash
pip install -r requirements.txt
pip install git+https://github.com/cbg-ethz/bmi.git
```

### 2. Basic Usage

**Run with unified model (default):**
```bash
python main.py --task_type multinormal --dim 3 --strength 200
```

**Run with MOE architecture:**
```bash
python main.py --use_moe --task_type multinormal --dim 3 --strength 200
```

**Specify GPU:**
```bash
export CUDA_VISIBLE_DEVICES=0
python main.py --use_moe --dim 10 --strength 1100
```

### 3. Architecture Selection

**Unified Model:**
```bash
python main.py --task_type multinormal --dim 10 --strength 1100
```

**MOE Model with custom SNR threshold:**
```bash
python main.py --use_moe --snr_threshold 3.0 --task_type multinormal --dim 10 --strength 1100
```

### 4. Training Configuration

**Custom training parameters:**
```bash
python main.py \
    --use_moe \
    --task_type spiral \
    --dim 20 \
    --strength 1550 \
    --max_epochs 750 \
    --learning_rate 1e-3 \
    --batch_size 256 \
    --use_ema
```

### 5. Checkpoint Management

**Load from checkpoint:**
```bash
python main.py --load_ckpt --use_moe
```

**Resume training with more epochs:**
```bash
python main.py --load_ckpt --max_epochs 1000
```

### 6. Batch Experiments

**Run all combinations with unified model:**
```bash
python main.py --run_all
```

**Run all combinations with MOE:**
```bash
python main.py --run_all --use_moe
```

This runs experiments across:
- **Dimensions**: 3, 5, 10, 20, 50
- **Strengths**: 200, 650, 1100, 1550, 2000  
- **Task types**: multinormal, half_cube, spiral

## Configuration Parameters

### Architecture Parameters
| Parameter | Description | Default | MOE Only |
|-----------|-------------|---------|----------|
| `--use_moe` | Enable MOE architecture | False | - |
| `--snr_threshold` | SNR threshold for model routing | 5.0 | ✓ |

### Task Parameters
| Parameter | Description | Options |
|-----------|-------------|---------|
| `--task_type` | Type of task | multinormal, half_cube, spiral |
| `--dim` | Task dimension | 3, 5, 10, 20, 50 |
| `--strength` | Task strength parameter | 200, 650, 1100, 1550, 2000 |

### Model Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_dim` | Hidden dimension for networks | 64 |
| `--logsnr_loc` | LogSNR distribution location | 2.0 |
| `--logsnr_scale` | LogSNR distribution scale | 3.0 |

### Training Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--learning_rate` | Learning rate | 1e-3 |
| `--max_epochs` | Maximum training epochs | 500 |
| `--batch_size` | Training batch size | 256 |
| `--use_ema` | Enable exponential moving average | False |
| `--ema_decay` | EMA decay rate | 0.999 |

### Data Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_sample_num` | Training samples | 100000 |
| `--test_num` | Test samples | 10000 |
| `--preprocess` | Data preprocessing | rescale |

## Project Structure

```
DMI/
├── MMG_Unet_MOE.py           # MOE architecture implementation
├── MMG_Unet.py               # Unified model implementation  
├── main.py                    # Main training and evaluation script
├── model/
│   └── denoiser.py           # Denoiser network architecture
├── utils/
│   └── utils.py              # Utility functions and EMA
├── checkpoints/              # Model checkpoints (auto-created)
├── lightning_logs/           # TensorBoard logs (auto-created)
├── results_moe_*.json        # MOE experiment results
├── results_unified_*.json    # Unified model results
└── requirements.txt          # Dependencies
```

## Core Methods

Both architectures share these key methods:

### Essential Functions
- **`nll()`**: Calculate negative log likelihood
- **`mse()`**: Calculate mean squared error with optional routing
- **`mse_orthogonal()`**: Calculate orthogonal MSE for MI estimation  
- **`estimate()`**: Estimate mutual information on test data
- **`noisy_channel()`**: Add noise based on log SNR
- **`logistic_integrate()`**: Generate integration points

### Training Features
- Mixed conditional/unconditional training
- Automatic checkpoint saving
- Periodic MI estimation during training
- TensorBoard logging with detailed metrics
- EMA support for training stability

## Results and Analysis

### Automatic Result Tracking
Results are saved to JSON files with detailed metrics:

```json
{
  "task": "multinormal_sparse_1100_dim_10",
  "gt_mi": 8.465,
  "mi_estimate": 8.431,
  "mi_estimate_orthogonal": 8.447,
  "estimator": "MMGEstimator_MOE",
  "use_moe": true,
  "snr_threshold": 5.0,
  "model_dim": 64,
  "learning_rate": 0.001,
  "max_epochs": 500
}
```

### Performance Comparison
| Architecture | Advantages | Best Use Cases |
|--------------|------------|----------------|
| **Unified** | Faster, simpler, fewer parameters | Quick experiments, baseline comparisons |
| **MOE** | Specialized expertise, potentially higher accuracy | High-precision requirements, research |

## Example Workflows

### Quick Comparison
```bash
# Train unified model
python main.py --task_type multinormal --dim 10 --strength 1100

# Train MOE model with same settings
python main.py --use_moe --task_type multinormal --dim 10 --strength 1100
```

### Research Experiment
```bash
# Comprehensive MOE evaluation
python main.py --run_all --use_moe --use_ema --max_epochs 750

# Compare with unified baseline
python main.py --run_all --use_ema --max_epochs 750
```

### High-Dimensional Task
```bash
python main.py \
    --use_moe \
    --task_type spiral \
    --dim 50 \
    --strength 2000 \
    --model_dim 128 \
    --max_epochs 1000 \
    --use_ema
```

## Tips for Best Results

### Architecture Selection
- **Use Unified** for: Quick experiments, limited compute, baseline comparisons
- **Use MOE** for: High-precision requirements, research comparisons, complex tasks

### Hyperparameter Guidelines  
- **Low dimensions (≤10)**: model_dim=64, epochs=500
- **High dimensions (>10)**: model_dim=128, epochs=750+
- **Complex tasks**: Enable EMA, increase epochs
- **MOE specific**: Adjust snr_threshold based on task characteristics

### Performance Optimization
- Enable `--use_ema` for more stable training
- Use larger `--model_dim` for complex high-dimensional tasks
- Adjust `--snr_threshold` (MOE only) if convergence issues occur
- Monitor TensorBoard logs for training progress

## Troubleshooting

### Common Issues
- **Import errors**: Ensure BMI package is installed
- **Memory issues**: Reduce `--batch_size` or `--model_dim`  
- **Convergence problems**: Enable `--use_ema`, adjust `--learning_rate`
- **MOE routing issues**: Adjust `--snr_threshold`
