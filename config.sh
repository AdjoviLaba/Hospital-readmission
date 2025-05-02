#!/bin/bash
# Configuration file for MIMIC-IV readmission prediction

# Path to MIMIC-IV dataset
MIMIC_IV_PATH="/mimic-iv"  

# Output directories
OUTPUT_DIR="./mimic_iv_output"
PROCESSED_DATA_DIR="$OUTPUT_DIR/processed_data"
MODEL_OUTPUT_DIR="$OUTPUT_DIR/model_output"

# Model parameters
DROPOUT=0.3
LEARNING_RATE=2e-5
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
NUM_EPOCHS=2
LOGGING_STEPS=500

# Data preparation parameters
READMISSION_WINDOW=30  # in days
SEED=42
TEST_SIZE=0.2  