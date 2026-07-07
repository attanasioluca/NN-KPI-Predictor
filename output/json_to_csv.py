import json
import os
 
files = [
    "output/metrics_lr_model.json",
    "output/metrics_simple_nn.json",
    "output/metrics_complex_nn.json",
    "output/metrics_deep_nn.json"
]
 
columns = [
    "Model",
    "Best Test Loss",
    "MSE - Cost",
    "MSE - Cycle Time",
    "MSE - Waiting Time",
    "Error - Cost",
    "Error - Cycle Time",
    "Error - Waiting Time",
    "Percentage - Cost",
    "Percentage - Cycle Time",
    "Percentage - Waiting Time"
]

print("\t".join(columns))

for file_name in files:
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue
        
    with open(file_name, 'r') as f:
        data = json.load(f)
        
    # Safely extract Best Test Loss
    test_loss = data.get("Best Test Loss", "")
    if test_loss != "":
        test_loss = f",{float(test_loss):.4f},"
        
    
    row = [
        data.get("model_name", "Unknown Model"),
        test_loss,
        f"{data['MSE']['cost']:.2f},",
        f"{data['MSE']['cycle_time']:.2f},",
        f"{data['MSE']['waiting_time']:.2f},",
        f"{data['Error']['cost']:.2f},",
        f"{data['Error']['cycle_time']:.2f},",
        f"{data['Error']['waiting_time']:.2f},",
        f"{data['Percentage']['cost']:.2f},",
        f"{data['Percentage']['cycle_time']:.2f},",
        f"{data['Percentage']['waiting_time']:.2f}"
    ]
    
    print("\t".join(row))