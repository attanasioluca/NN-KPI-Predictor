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

class PharmacySurrogate(nn.Module):
    def __init__(self, input_size, dropout_rate=0.1):
        super(PharmacySurrogate, self).__init__()
        self.shared_entry = nn.Sequential(nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(dropout_rate))
        self.shared_h1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(dropout_rate)
        self.shared_h2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(dropout_rate)
        self.shared_out = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Mish())

        # 3 Branches x 1 Output = 3 Targets total
        self.total_cost_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.cycle_time_branch   = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))
        self.waiting_time_branch = nn.Sequential(nn.Linear(64, 32), nn.Mish(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.shared_entry(x)
        h1 = self.drop1(self.act1(self.bn1(self.shared_h1(x))))
        h2 = self.drop2(self.act2(self.bn2(self.shared_h2(h1))))
        h2 = h2 + h1 
        shared_features = self.shared_out(h2)
        return torch.cat((self.total_cost_branch(shared_features), self.cycle_time_branch(shared_features), self.waiting_time_branch(shared_features)), dim=1) 

DATA_FILE = "data/real/sim_data_waiting_times.csv" 
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(DATA_FILE)
X_df = df.drop(columns=["scenario_id", "kpi_total_cost", "kpi_std_total_cost", "kpi_cycle_time", "kpi_std_cycle_time", "kpi_waiting_time", "kpi_std_waiting_time", "n_reps_used", "converged","converged_wait","converged_cost","converged_duration"])
y_df = df[["kpi_total_cost", "kpi_cycle_time", "kpi_waiting_time"]]

# APPLY LOG TRANSFORMATION TO TARGETS
y_log = np.log1p(y_df.values)

input_size = X_df.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X_df.values, y_log, test_size=0.20, random_state=42)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    
    train_loader = DataLoader(SimulationDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SimulationDataset(X_test_scaled, y_test_scaled), batch_size=batch_size, shuffle=False)

    model = PharmacySurrogate(input_size, dropout_rate=dropout_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() 
    
    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # Standard Loss for the 3 KPIs
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_kpi_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                val_kpi_loss += criterion(model(batch_X), batch_y).item() * batch_X.size(0)
                
        val_kpi_loss /= len(test_loader.dataset)
        trial.report(val_kpi_loss, epoch)
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

    best_params = study.best_trial.params
    
    print("\n" + "="*50)
    print("🚀 COPY & PASTE INTO hypertuned_model.py")
    print("="*50)
    print(f"BATCH_SIZE = {best_params['batch_size']}")
    print(f"LEARNING_RATE = {best_params['lr']}")
    print(f"WEIGHT_DECAY = {best_params['weight_decay']}")
    print(f"DROPOUT_RATE = {best_params['dropout_rate']}")
    print("="*50 + "\n")

if __name__ == "__main__": 
    main()