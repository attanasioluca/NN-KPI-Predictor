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

class SimulationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x): return x + self.net(x) 

class DeepPharmacySurrogate(nn.Module):
    def __init__(self, input_size, hidden_dim=256, num_blocks=4, dropout_rate=0.1):
        super(DeepPharmacySurrogate, self).__init__()
        
        self.entry = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)])
        
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.Mish()
        )

        def make_branch():
            return nn.Sequential(
                nn.Linear(128, 64), 
                nn.LayerNorm(64), 
                nn.Mish(),

                nn.Linear(64, 32), 
                nn.Mish(),
                
                nn.Linear(32, 1)
            )
            
        self.total_cost_branch   = make_branch()
        self.cycle_time_branch   = make_branch()
        self.waiting_time_branch = make_branch()

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        shared = self.shared_out(x)
        return torch.cat((
            self.total_cost_branch(shared), 
            self.cycle_time_branch(shared), 
            self.waiting_time_branch(shared)
        ), dim=1) 

def main():
    DATA_FILE = "data/real/sim_data_waiting_times.csv" 
    EPOCHS = 10000
   
    # --- PLUG IN YOUR OPTUNA RESULTS HERE ---
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0004365446315622289
    WEIGHT_DECAY = 0.0011832421845475193
    HIDDEN_DIM = 512
    NUM_BLOCKS = 6
    DROPOUT_RATE = 0.3811733811859317
    # ----------------------------------------
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # --- A. Data Loading & Preprocessing ---
    print("Loading dataset...")
    # 1. Load the ENTIRE dataset (remove nrows)
    df = pd.read_csv(DATA_FILE)

    # Note: Preserving your extended drop list for this specific script
    X_df = df.drop(columns=["scenario_id", "kpi_total_cost", "kpi_std_total_cost", "kpi_cycle_time", "kpi_std_cycle_time", "kpi_waiting_time", "kpi_std_waiting_time",  "n_reps_used", "converged","converged_wait","converged_cost","converged_duration"])
    y_df = df[["kpi_total_cost", "kpi_cycle_time", "kpi_waiting_time"]]
    
    input_size = X_df.shape[1]
    output_size = y_df.shape[1]
    print(f"Features: {input_size} | Targets: {output_size}")
    
    # 2. Split the FULL dataset once to create a universal, locked test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_df.values, y_df.values, test_size=0.20, random_state=42
    )

    TRAIN_SAMPLES = 5000 
    
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

    os.makedirs("complex_model/output", exist_ok=True)
    joblib.dump(x_scaler, 'complex_model/output/x_scaler.pkl')
    joblib.dump(y_scaler, 'complex_model/output/y_scaler.pkl')

    train_dataset = SimulationDataset(X_train_scaled, y_train_scaled)
    test_dataset = SimulationDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepPharmacySurrogate(input_size, HIDDEN_DIM, NUM_BLOCKS, DROPOUT_RATE).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    best_test_loss = float('inf')
    patience = 200
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            
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
            torch.save(model.state_dict(), "deep_network/output/surrogate_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
            break 

    print(f"Training finished in {time.time() - start_time:.2f}s. Saved Deep model.")

    # ==========================================
    # FINAL EVALUATION & JSON EXPORT
    # ==========================================
    print("\nEvaluating BEST model for JSON export...")
    
    model.load_state_dict(torch.load("deep_network/output/surrogate_model.pth"))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    predictions_scaled = np.vstack(all_preds)
    y_test_scaled_eval = np.vstack(all_targets)
    
    test_mse_scaled = np.mean((predictions_scaled - y_test_scaled_eval) ** 2, axis=0)
    test_rmse_scaled = np.sqrt(test_mse_scaled)
    
    error_kpi = test_rmse_scaled * y_scaler.scale_                  
    percentage_kpi = (error_kpi / np.abs(y_scaler.mean_)) * 100     
    mse_kpi = error_kpi ** 2                                        
    
    metrics = {
        "model_name": "Deep Network - Hypertuned",
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

    with open("output/metrics_deep_nn.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Exported metrics to output/metrics_deep_nn.json")

if __name__ == "__main__":
    main()