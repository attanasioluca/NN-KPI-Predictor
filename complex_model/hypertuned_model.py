import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os
import json

# ==========================================
# 1. PYTORCH DATASET DEFINITION
# ==========================================
class SimulationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class PharmacySurrogate(nn.Module):
    def __init__(self, input_size, DROPOUT_RATE):
        super(PharmacySurrogate, self).__init__()
        
        self.shared_entry = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        self.shared_h1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(DROPOUT_RATE)
        
        self.shared_h2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(DROPOUT_RATE)
        
        self.shared_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish()
        )

        # 3 Output Branches
        self.total_cost_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.cycle_time_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.waiting_time_branch = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        h1 = self.drop1(self.act1(self.bn1(self.shared_h1(x))))
        h2 = self.drop2(self.act2(self.bn2(self.shared_h2(h1))))
        h2 = h2 + h1 # Residual Connection
        
        shared_features = self.shared_out(h2)
        
        t_cost = self.total_cost_branch(shared_features)
        cycle  = self.cycle_time_branch(shared_features)
        wait   = self.waiting_time_branch(shared_features)
        
        return torch.cat((t_cost, cycle, wait), dim=1) 

# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================
def main():
    DATA_FILE = "data/BIMP/sim_data_waiting_times.csv" 
    EPOCHS = 10000

    BATCH_SIZE = 256
    LEARNING_RATE = 0.0004970082133946468
    WEIGHT_DECAY = 2.217485788658681e-06
    DROPOUT_RATE = 0.41629303761709978
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # --- A. Data Loading & Preprocessing ---
    print("Loading dataset...")
    # 1. Load the ENTIRE dataset (remove nrows)
    df = pd.read_csv(DATA_FILE)

   # hypertuned_model.py (Data Loading Phase)
    # ...
    X_df = df.drop(columns=["scenario_id", "kpi_total_cost", "kpi_std_total_cost", "kpi_cycle_time", "kpi_std_cycle_time", "kpi_waiting_time", "kpi_std_waiting_time", "n_reps_used", "converged","converged_wait","converged_cost","converged_duration"])
    y_df = df[["kpi_total_cost", "kpi_cycle_time", "kpi_waiting_time" ]]
    
    # APPLY LOG TRANSFORMATION TO TARGETS
    y_log = np.log1p(y_df.values)
    
    input_size = X_df.shape[1]
    output_size = y_df.shape[1]
    print(f"Features: {input_size} | Targets: {output_size}")

    # 2. Split using y_log instead of y_df.values
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_df.values, y_log, test_size=0.20, random_state=42
    )

    TRAIN_SAMPLES = 10000 
    
    # 4. Slice the training arrays down to the desired size
    X_train = X_train_full[:TRAIN_SAMPLES]
    y_train = y_train_full[:TRAIN_SAMPLES]
    
    print(f"--> Training on {len(X_train)} samples.")
    print(f"--> Testing on {len(X_test)} consistent samples.")

    print("Scaling data...")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # 5. Fit scalers ONLY on the active training subset
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Ensure output directory exists before saving scalers or models
    os.makedirs("complex_model/output", exist_ok=True)
    
    joblib.dump(x_scaler, 'complex_model/output/x_scaler.pkl')
    joblib.dump(y_scaler, 'complex_model/output/y_scaler.pkl')

    train_dataset = SimulationDataset(X_train_scaled, y_train_scaled)
    test_dataset = SimulationDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PharmacySurrogate(input_size, DROPOUT_RATE=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    best_test_loss = float('inf')
    patience = 450
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # Standard Loss 
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                
        test_loss /= len(test_loader.dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), "complex_model/output/surrogate_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
            break 

    print(f"Training finished in {time.time() - start_time:.2f}s. Saved 3-output model.")

    # ==========================================
    # FINAL EVALUATION & JSON EXPORT
    # ==========================================
    print("\nEvaluating BEST model for JSON export...")
    
    model.load_state_dict(torch.load("complex_model/output/surrogate_model.pth"))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    # hypertuned_model.py (Final Evaluation Phase)
    predictions_scaled = np.vstack(all_preds)
    y_test_scaled_eval = np.vstack(all_targets)
    
    # 1. Reverse the StandardScaler
    predictions_log = y_scaler.inverse_transform(predictions_scaled)
    y_test_log = y_scaler.inverse_transform(y_test_scaled_eval)
    
    # 2. Reverse the Log Transformation (expm1)
    predictions_true = np.expm1(predictions_log)
    y_test_true = np.expm1(y_test_log)
    
    # 3. Calculate True Errors
    error_kpi = np.sqrt(np.mean((predictions_true - y_test_true) ** 2, axis=0))
    mean_true = np.mean(y_test_true, axis=0)
    percentage_kpi = (error_kpi / np.abs(mean_true)) * 100     
    mse_kpi = error_kpi ** 2                                       
    
    metrics = {
        "model_name": "Complex Model - Hypertuned",
        "Best Test Loss":float(best_test_loss),
        "MSE": {
            "cost": float(mse_kpi[0]), 
            "cycle_time": float(mse_kpi[1]), 
            "waiting_time": float(mse_kpi[2])
        },
        "Error": {
            "cost": float(error_kpi[0]), 
            "cycle_time": float(error_kpi[1]), 
            "waiting_time": float(error_kpi[2])
        },
        "Percentage": {
            "cost": float(percentage_kpi[0]), 
            "cycle_time": float(percentage_kpi[1]), 
            "waiting_time": float(percentage_kpi[2])
        }
    }

    with open("output/metrics_complex_nn.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Exported metrics to /output/metrics_complex_nn.json")

if __name__ == "__main__":
    main()