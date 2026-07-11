import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import optuna

class SimulationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- DEEP RESIDUAL BLOCK ---
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
    def forward(self, x):
        return x + self.net(x) # Skip connection

# --- DEEP ARCHITECTURE ---
class DeepPharmacySurrogate(nn.Module):
    def __init__(self, input_size, hidden_dim=256, num_blocks=4, dropout_rate=0.1):
        super(DeepPharmacySurrogate, self).__init__()
        
        # Initial Projection
        self.entry = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        
        # Deep Residual Trunk
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)])
        
        # WIDENED BOTTLENECK: Preserving hidden_dim instead of compressing to 128
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )

        # Deeper Branches for complex KPI mapping
        def make_branch():
            return nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.Mish(),
                nn.Linear(64, 32), nn.Mish(),
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

DATA_FILE = "data/real/sim_data_waiting_times.csv" 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

df = pd.read_csv(DATA_FILE)
X_df = df.drop(columns=["scenario_id", "kpi_total_cost", "kpi_std_total_cost", "kpi_cycle_time", "kpi_std_cycle_time", "kpi_waiting_time", "kpi_std_waiting_time", "n_reps_used", "converged","converged_wait","converged_cost","converged_duration"])
y_df = df[["kpi_total_cost", "kpi_cycle_time", "kpi_waiting_time"]]

input_size = X_df.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X_df.values, y_df.values, test_size=0.20, random_state=42)

x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

def objective(trial):
    # CONSTRAINED SEARCH SPACE
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.45)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512])
    num_blocks = trial.suggest_int("num_blocks", 4, 8)
    
    train_loader = DataLoader(SimulationDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SimulationDataset(X_test_scaled, y_test_scaled), batch_size=batch_size, shuffle=False)

    model = DeepPharmacySurrogate(input_size, hidden_dim, num_blocks, dropout_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # LEARNING RATE SCHEDULER
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    criterion = nn.MSELoss() 
    
    # INCREASED EPOCHS
    for epoch in range(300):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        
        # Step the scheduler at the end of every epoch
        scheduler.step()
            
        model.eval()
        val_kpi_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                val_kpi_loss += criterion(model(batch_X), batch_y).item() * batch_X.size(0)
                
        val_kpi_loss /= len(test_loader.dataset)
        trial.report(val_kpi_loss, epoch)
        
        # Pruning happens based on intermediate values reported above
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    return val_kpi_loss

def main():
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15), 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=100, timeout=36000)
    
    print(f"\nBest Trial Validation MSE: {study.best_trial.value:.5f}")
    
    # --- Generate Complete Copy-Pasteable Block ---
    best_params = study.best_trial.params
    
    print("\n" + "="*50)
    print("Copy and paste into hypertuned_model.py")
    print("="*50)
    print(f"BATCH_SIZE = {best_params['batch_size']}")
    print(f"LEARNING_RATE = {best_params['lr']}")
    print(f"WEIGHT_DECAY = {best_params['weight_decay']}")
    print(f"HIDDEN_DIM = {best_params['hidden_dim']}")
    print(f"NUM_BLOCKS = {best_params['num_blocks']}")
    print(f"DROPOUT_RATE = {best_params['dropout_rate']}")
    print("="*50 + "\n")

if __name__ == "__main__": 
    main()
