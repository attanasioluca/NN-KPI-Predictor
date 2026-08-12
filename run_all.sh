#!/bin/bash

# Exit script immediately if any command fails
set -e

echo "=================================================="
echo " Starting Full NN-KPI-Predictor Benchmark Test"
echo "=================================================="

# 1. LR Model
echo -e "\n[1/12] Running LR Model (synthetic)..."
python3 models/LR_model/lr_model.py synthetic

echo -e "\n[2/12] Running LR Model (BIMP)..."
python3 models/LR_model/lr_model.py BIMP

# 2. Simple Model
echo -e "\n[3/12] Running Simple Model (synthetic)..."
python3 models/simple_model/model.py synthetic

echo -e "\n[4/12] Running Simple Model (BIMP)..."
python3 models/simple_model/model.py BIMP

# 3. Complex Model
echo -e "\n[5/12] Hypertuning Complex Model (synthetic)..."
python3 models/complex_model/hypertuner.py synthetic

echo -e "\n[6/12] Training Hypertuned Complex Model (synthetic)..."
python3 models/complex_model/hypertuned_model.py synthetic

echo -e "\n[7/12] Hypertuning Complex Model (BIMP)..."
python3 models/complex_model/hypertuner.py BIMP

echo -e "\n[8/12] Training Hypertuned Complex Model (BIMP)..."
python3 models/complex_model/hypertuned_model.py BIMP

# 4. Deep Network
echo -e "\n[9/12] Hypertuning Deep Network (synthetic)..."
python3 models/deep_network/hypertuner.py synthetic

echo -e "\n[10/12] Training Hypertuned Deep Network (synthetic)..."
python3 models/deep_network/hypertuned_model.py synthetic

echo -e "\n[11/12] Hypertuning Deep Network (BIMP)..."
python3 models/deep_network/hypertuner.py BIMP

echo -e "\n[12/12] Training Hypertuned Deep Network (BIMP)..."
python3 models/deep_network/hypertuned_model.py BIMP

echo -e "\n=================================================="
echo "  All 12 Benchmark Runs Completed Successfully!"
echo "=================================================="

python3 models/output/json_to_csv.py

