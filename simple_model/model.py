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
    def __init__(self, input_size):
        super(PharmacySurrogate, self).__init__()
        
        self.shared_entry = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.Mish()
        )
        self.shared_out = nn.Sequential(nn.Linear(64, 32), nn.Mish())

        # Branch 1: Total Cost
        self.cost_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))
        # Branch 2: Cycle Time
        self.cycle_time_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))
        # Branch 3: Waiting Time
        self.waiting_time_branch = nn.Sequential(nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        shared_features = self.shared_out(x)
        
        t_cost = self.cost_branch(shared_features)
        cycle = self.cycle_time_branch(shared_features)
        wait = self.waiting_time_branch(shared_features)

        return torch.cat((t_cost, cycle, wait), dim=1)

# ==========================================
# 2b. TARGET TRANSFORM HELPERS
# ==========================================
# kpi_cycle_time and kpi_waiting_time are heavily right-skewed (skew ~35-37,
# a handful of rows sit 50+ standard deviations from the mean -- these are
# scenarios where arrival rate approaches server capacity and wait times
# blow up nonlinearly, classic queueing-theory saturation behavior).
# log1p compresses that tail before StandardScaler ever sees it, so a few
# extreme scenarios don't dominate the MSE loss.
TARGET_COLS = ["kpi_total_cost", "kpi_cycle_time", "kpi_waiting_time"]
LOG_TRANSFORM_COLS = ["kpi_cycle_time", "kpi_waiting_time"]
LOG_COL_IDX = [TARGET_COLS.index(c) for c in LOG_TRANSFORM_COLS]


def inverse_transform_targets(y_scaled, y_scaler):
    """Undo StandardScaler, then undo log1p on the columns that were logged.
    Returns predictions/targets in real-world units (dollars, seconds)."""
    y_unscaled = y_scaler.inverse_transform(y_scaled)
    y_real = y_unscaled.copy()
    y_real[:, LOG_COL_IDX] = np.expm1(y_real[:, LOG_COL_IDX])
    return y_real


# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================
def main():
    DATA_FILE = "data/synthetic/sim_data_waiting_times.csv"
    BATCH_SIZE = 64
    EPOCHS = 2500
    
    LEARNING_RATE = 0.5e-3
    
    # Check for GPU (Apple Silicon MPS or Nvidia CUDA)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available(): 
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    print("Loading dataset...")
    # 1. Load the ENTIRE dataset (remove nrows)
    df = pd.read_csv(DATA_FILE) 

    # 1b. Drop simulation runs that didn't converge -- their KPI estimates
    # aren't reliable ground truth, so they shouldn't be training targets.
    conv_flags = ["converged", "converged_wait", "converged_cost", "converged_duration"]
    n_before = len(df)
    df = df[df[conv_flags].all(axis=1)].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} unconverged rows ({len(df)} remain).")

    X_df = df.drop(columns=["scenario_id", "kpi_total_cost", "kpi_std_total_cost", "kpi_cycle_time", "kpi_std_cycle_time", "kpi_waiting_time", "kpi_std_waiting_time", "n_reps_used", "converged","converged_wait","converged_cost","converged_duration"])
    y_df = df[TARGET_COLS]

    input_size = X_df.shape[1]
    output_size = y_df.shape[1]
    print(f"Features: {input_size} | Targets: {output_size}")

    # 1c. Log1p the heavy-tailed KPIs so extreme (but real) outlier scenarios
    # don't dominate the MSE loss once everything is standardized.
    y_raw = y_df.values.astype(np.float64)
    y_log = y_raw.copy()
    y_log[:, LOG_COL_IDX] = np.log1p(y_log[:, LOG_COL_IDX])

    # 2. Split the FULL dataset once to create a universal, locked test set.
    # Split X, the log-transformed y (used for training), and the raw y
    # (used later to report real-world-unit metrics) together so indices
    # stay aligned.
    X_train_full, X_test, y_train_full, y_test, _, y_test_raw = train_test_split(
        X_df.values, y_log, y_raw, test_size=0.2, random_state=42
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

    # Save the scalers
    joblib.dump(x_scaler, 'simple_model/output/x_scaler.pkl')
    joblib.dump(y_scaler, 'simple_model/output/y_scaler.pkl')
    print("Saved x_scaler.pkl and y_scaler.pkl")

    # --- B. DataLoader Setup ---
    train_dataset = SimulationDataset(X_train_scaled, y_train_scaled)
    test_dataset = SimulationDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- C. Model Initialization ---
    model = PharmacySurrogate(input_size).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"Starting training for up to {EPOCHS} epochs...")
    start_time = time.time()
    
    best_test_loss = float('inf')
    patience = 200  
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()  # Updates learning rate
        
        # --- E. Validation/Testing Loop ---
        model.eval()
        test_loss = 0.0
        test_preds_scaled = []
        test_targets_scaled = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                
                # Standard scalar loss for early stopping (still in log+scaled
                # space -- this is what actually drives optimization)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item() * batch_X.size(0)

                test_preds_scaled.append(predictions.cpu().numpy())
                test_targets_scaled.append(batch_y.cpu().numpy())
                
        test_loss /= len(test_loader.dataset)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            preds_scaled = np.vstack(test_preds_scaled)

            # Undo StandardScaler + log1p to get back to real-world units
            # (dollars, seconds). NOTE: this is no longer a simple
            # scale_-multiplication trick -- expm1 is nonlinear, so we
            # reconstruct real values directly from predictions and compare
            # against the untransformed y_test_raw.
            preds_real = inverse_transform_targets(preds_scaled, y_scaler)
            rmse_raw = np.sqrt(np.mean((preds_real - y_test_raw) ** 2, axis=0))
            rmse_pct = (rmse_raw / np.abs(y_test_raw.mean(axis=0))) * 100
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss**2:.4f} | Test Loss: {test_loss**2:.4f}")
            print(f"   ↳ Test RMSE -> Cost: ${rmse_raw[0]:.2f} (±{rmse_pct[0]:.1f}%) | "
                  f"Cycle Time: {rmse_raw[1]:.2f} (±{rmse_pct[1]:.1f}%) | "
                  f"Waiting Time: {rmse_raw[2]:.1f}s (±{rmse_pct[2]:.1f}%)")

        # --- EARLY STOPPING LOGIC ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Overwrite the saved model ONLY when test loss reaches a new low
            torch.save(model.state_dict(), "simple_model/output/surrogate_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
            print(f"Best Test Loss achieved: {best_test_loss**2:.4f}")
            break 

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f} seconds.")
    print("Saved the BEST trained model weights to surrogate_model.pth")

    # ==========================================
    # FINAL EVALUATION & JSON EXPORT
    # ==========================================
    print("\nEvaluating BEST model for JSON export...")
    
    # Loads the best weights
    model.load_state_dict(torch.load("simple_model/output/surrogate_model.pth"))
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
    predictions_real = inverse_transform_targets(predictions_scaled, y_scaler)

    error_kpi = np.sqrt(np.mean((predictions_real - y_test_raw) ** 2, axis=0))      # RMSE in real units
    percentage_kpi = (error_kpi / np.abs(y_test_raw.mean(axis=0))) * 100            # error / avg * 100
    mse_kpi = error_kpi ** 2                                                        # MSE in real terms
    
    import json
    metrics = {
        "model_name": "Simple NN",
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
    
    with open("output/metrics_simple_nn.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()