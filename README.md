# Hospital Readmission Prediction

This repository contains an implementation of a hospital readmission prediction model using clinical notes from the MIMIC-IV dataset. The model is based on the paper "Compositional and Hierarchical Semantic Learning Model for Hospital Readmission Prediction" by Gao et al., adapted to work with the newer MIMIC-IV dataset.

## Overview

The project implements a predictive model for hospital readmission using clinical notes from electronic health records. The current implementation focuses on the pure BERT component, with plans to extend to the full graph-based model in future work.

### Features

- BERT-based encoding of clinical notes
- Support for discharge summaries and early admission notes
- Binary classification for 30-day readmission prediction
- Option to run on CPU or GPU

## Dataset

The model is designed to work with the MIMIC-IV dataset. Due to licensing restrictions, the dataset cannot be included directly in this repository. However, you can download the preprocessed data from the following link:

[Download Preprocessed MIMIC-IV Data (Google Drive)](https://drive.google.com/drive/folders/your-folder-id)

Alternatively, you can prepare your own data following these steps:
1. Obtain access to the MIMIC-IV dataset through [PhysioNet](https://physionet.org/content/mimiciv/2.1/)
2. Preprocess the data using the scripts in the `preprocessing/` directory
3. Place the processed CSV files in the `processed_data/` directory

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AdjoviLaba/Hospital-readmission.git
cd hospital-readmission-prediction
```

2. Create a virtual environment and install dependencies:
```bash
# Create a virtual environment
python -m venv chslm
source chslm/bin/activate  # On Windows: chslm\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. Download the preprocessed data and place it in the `processed_data/` directory

## Running the Model

### Pure BERT Mode (CPU)

```bash
python run.py \
  --output_dir ./model_output \
  --dropout 0.3 \
  --learning_rate 2e-5 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --num_classes 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --logging_steps 500 \
  --dataset_folder ./processed_data/ \
  --pure_bert \
  --bert_model_dir emilyalsentzer/Bio_ClinicalBERT \
  --cuda_id cpu
```

### Pure BERT Mode (GPU)

If you have a compatible GPU, you can run the model on GPU by changing the `--cuda_id` parameter:

```bash
python run.py \
  --output_dir ./model_output \
  --dropout 0.3 \
  --learning_rate 2e-5 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --num_classes 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --logging_steps 500 \
  --dataset_folder ./processed_data/ \
  --pure_bert \
  --bert_model_dir emilyalsentzer/Bio_ClinicalBERT \
  --cuda_id cuda
```

Note: If you have multiple GPUs, you can specify a specific GPU by using `cuda:0`, `cuda:1`, etc.

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output_dir` | Directory to store output files | `./model_output` |
| `--dataset_folder` | Directory containing the dataset files | `./processed_data/` |
| `--bert_model_dir` | Path or name of the pre-trained BERT model | `emilyalsentzer/Bio_ClinicalBERT` |
| `--num_classes` | Number of classes for classification | `1` |
| `--cuda_id` | GPU ID to use (use 'cpu' for CPU-only) | `cuda` |
| `--seed` | Random seed for initialization | `2019` |
| `--pure_bert` | Use pure BERT model without graph components | - |
| `--dropout` | Dropout rate for model | `0.3` |
| `--train_batch_size` | Batch size for training | `8` |
| `--eval_batch_size` | Batch size for evaluation | `8` |
| `--gradient_accumulation_steps` | Number of updates steps to accumulate before performing a backward/update pass | `1` |
| `--learning_rate` | Learning rate for optimizer | `2e-5` |
| `--max_seq_length` | Maximum sequence length for BERT | `512` |
| `--num_train_epochs` | Number of training epochs | `2` |
| `--logging_steps` | Log every X updates steps | `500` |

## Project Structure

- `model.py`: Contains the model implementation (CHSLM and MIMIC_Bert_Only)
- `datasets.py`: Dataset processing and loading
- `trainer.py`: Training and evaluation functionality
- `run.py`: Main script for running the model
- `modeling_readmission.py`: Custom BERT implementation for readmission prediction
- `self_attention.py`: Self-attention module for the model
- `attentiveFP.py`: AttentiveFP implementation for graph neural networks
- `processed_data/`: Directory for preprocessed data files
- `model_output/`: Directory for output files and trained models

## Requirements

- Python 3.7+
- PyTorch 1.10.0+
- Transformers 4.18.0+
- PyTorch Geometric 2.0.4+
- NLTK
- SpaCy
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## Future Work

1. Implementation of the full graph-based model with heterogeneous GNN
2. GPU compatibility fixes for RTX 30xx series
3. Multi-modal data integration
4. Improved preprocessing for MIMIC-IV specific features



## License

This project is licensed under the MIT License - see the LICENSE file for details.