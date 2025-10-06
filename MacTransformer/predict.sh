#!/bin/bash

# Path to your input SMILES text file
INPUT_FILE="/home/macrocycles/drug_sinter/input_smiles.txt"

# Activate conda env (if needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv  # replace 'myenv' with your environment name

# Run prediction
python /home/macrocycles/MacTransformer/predict.py "$INPUT_FILE"
