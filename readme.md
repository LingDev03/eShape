
# Efficient Shapelet-Based Time Series Classification

This project implements an efficient and interpretable method for time series classification using **position-aware shapelet discovery**. It improves traditional shapelet-based techniques by incorporating position awareness and learnable shapelet representations using PyTorch.

---

## Directory Structure

```
└── eShape/
    ├── model/
    │   ├── shapelet_discovery.py                # Main shapelet discovery logic
    │   └── position_shapelet.py                 # Position-aware enhancements
    │
    ├── util/
    │   ├── auto_pisd.py                         # Auto shapelet configuration
    │   ├── loading_bar.py                       # Progress bar utility
    │   ├── log.py                               # Logging utility
    │   ├── pst_support_method.py                # Support methods for position-shapelet
    │   ├── shapelet_support_method.py           # Shapelet-related utilities
    │   └── ucr_data_loader.py                   # UCR dataset loader
    │
    ├── dataset/
    │   └── UCRArchive_2018/                     # Dataset folder 
    │       ├── ECGFiveDays/                     
    │       │   ├── ECGFiveDays_TRAIN.tsv
    │       │   └── ECGFiveDays_TEST.tsv
    │       └── ...
    ├── results/                                 # Output directory for results
    ├── config.csv                               # Configuration file for experiment parameters
    ├── eShape.py                                # Main script to run the model
    └── requirements.txt                         # Python dependencies list
```

---

## Python Version

This project requires **Python 3.8**.

---

## Dataset Setup

Before running the experiments, you must download the UCR Time Series Classification Archive.

1. Download the dataset from: [UCR Archive 2018 (Google Drive)](https://drive.google.com/file/d/1fYP4f3FTlwfAMY5icE--AXljm5anDTSV/view?usp=sharing)
2. Extract the downloaded archive
3. Place the `UCRArchive_2018` folder inside `eShape/dataset/`

The final structure should be:
```
eShape/dataset/UCRArchive_2018/
    ├── ECGFiveDays/
    ├── PigAirwayPressure/
    ├── FordA/
    └── ... (other datasets)
```

---

## Installation

Install required dependencies using:

```bash
pip install -r requirements.txt
```
If using GPU with CUDA 11.3:

```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu113
```

---

## How to Run



```bash
python eShape.py \
  --dataset_name UWaveGestureLibraryAll \
  --num_shapelet 2 \
  --subset_ratio 0.2 \
  --r 4 \
  --epochs 1000
```
```bash
python eShape.py \
  --dataset_name StarlightCurves \
  --num_shapelet 1 \
  --subset_ratio 0.2 \
  --r 4 \
  --epochs 1000
```

```bash
python eShape.py \
  --dataset_name HandOutlines \
  --num_shapelet 0.05 \
  --subset_ratio 0.2 \
  --r 4 \
  --epochs 1000
```

```bash
python eShape.py \
  --dataset_name FordA \
  --num_shapelet 5 \
  --subset_ratio 0.2 \
  --r 4 \
  --epochs 1000
```bash
python eShape.py \
  --dataset_name PigAirwayPressure \
  --num_shapelet 10 \
  --subset_ratio 0.2 \
  --r 4 \
  --epochs 1000
```

```bash
python eShape.py \
  --dataset_name PigArtPressure \
  --num_shapelet 10 \
  --subset_ratio 0.2 \
  --r 4 \
  --epochs 1000
```

```bash
python eShape.py \
  --dataset_name PigCVP \
  --num_shapelet 10 \
  --subset_ratio 0.2 \
  --r 4 \
  --epochs 1000
```
```

Alternatively, define multiple experiment configurations in `config.csv` for batch processing.

---

## Key Arguments

| Argument           | Description |
|--------------------|-------------|
| `--momentum`       | SGD momentum (default: 0.9) |
| `--weight-decay`   | Weight decay regularization (default: 1e-5) |
| `--lr`             | Learning rate for optimizer (default: 1e-2) |
| `--dataset_name`   | Name of the dataset (e.g., ECGFiveDays) |
| `--num_shapelet`   | Proportion of top shapelets to select (e.g., 0.2 = 20%) |
| `--num_pip`        | Proportion of points for candidate shapelet extraction (e.g., 0.3 = 30%) |
| `--sge`            | Stop-gradient epochs — number of epochs where gradients are not backpropagated |
| `--processes`      | Number of parallel processes for shapelet extraction |
| `--bounding_norm`  | Bounding normalization constant (fixed at 100) |
| `--max_acc`        | Placeholder to track maximum accuracy (default: 0.0) |
| `--batch_size`     | Batch size used in training and validation loop (default: 16) |
| `--epochs`         | Total number of training epochs (default: 200) |
| `--threads`        | Number of CPU threads used by data loaders (default: 2) |
| `--smoothing`      | Label smoothing factor for classification loss (default: 0.1) |
| `--device`         | Device used for training (`cuda:0`, `cuda:1`, or `cpu`) |
| `--sep`            | Used in dataset pre-processing or loading (default: 1) |
| `--r`              | Shapelet expansion factor (e.g., 4x) |
| `--subset_ratio`   | Ratio of training data used in round-1 shapelet filtering (e.g., 0.2 = 20%) |

---

## Notes

- For GPU training, use `--device cuda:0`; otherwise use `--device cpu`.
- To disable label smoothing, set `smoothing` to 0.
- Ensure that `UCRArchive_2018` is properly placed under `eShape/dataset/` before running.

---
