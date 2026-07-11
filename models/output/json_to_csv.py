import json
import os

files = [
    "models/output/metrics_lr_model.json",
    "models/output/metrics_simple_nn.json",
    "models/output/metrics_complex_nn.json",
    "models/output/metrics_deep_nn.json"
]

columns = [
    "Data Source",
    "Model Name",
    "Best Test Loss",
    "MedAE - Cost",
    "MedAE - Cycle Time",
    "MedAE - Waiting Time",
    "MAE - Cost",
    "MAE - Cycle Time",
    "MAE - Waiting Time",
    "MedAE % - Cost",
    "MedAE % - Cycle Time",
    "MedAE % - Waiting Time"
]

print(",".join(columns))

for file_name in files:
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue
        
    with open(file_name, 'r') as f:
        # The new JSON format saves as a list to support multiple data sources
        data_list = json.load(f)
        
    # Fallback in case an older single-dict JSON is still present
    if isinstance(data_list, dict):
        data_list = [data_list]
        
    for data in data_list:
        # Safely extract Best Test Loss
        test_loss = data.get("Best Test Loss", "")
        if test_loss != "":
            test_loss = f"{float(test_loss):.4f}"
            
        row = [
            str(data.get("Data Source", "Unknown")),
            str(data.get("Model Name", "Unknown Model")),
            test_loss,
            f"{data['MedAE']['cost']:.2f}",
            f"{data['MedAE']['cycle_time']:.2f}",
            f"{data['MedAE']['waiting_time']:.2f}",
            f"{data['MAE']['cost']:.2f}",
            f"{data['MAE']['cycle_time']:.2f}",
            f"{data['MAE']['waiting_time']:.2f}",
            f"{data['MedAE_Percentage']['cost']:.2f}",
            f"{data['MedAE_Percentage']['cycle_time']:.2f}",
            f"{data['MedAE_Percentage']['waiting_time']:.2f}"
        ]
        
        print(",".join(row))