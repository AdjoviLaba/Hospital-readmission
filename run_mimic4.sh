#!/bin/bash

# Load configuration
source config.sh

# Update MIMIC_IV_PATH with the value from the environment if provided
if [ ! -z "$MIMIC_IV" ]; then
    echo "Using MIMIC_IV path from environment: $MIMIC_IV"
    MIMIC_IV_PATH=$MIMIC_IV
fi