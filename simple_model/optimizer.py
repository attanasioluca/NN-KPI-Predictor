import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd
import sys
import os

# 1. Get the absolute path of the parent directory (NN-KPI-Predictor)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. Add it to the system path if it isn't already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from simple_model.model import PharmacySurrogate 
from helpers.simulator import ScenarioSimulator

def evaluate_scenario(scenario_data, full_model, process_details, num_reps=100):
    simulator = ScenarioSimulator(scenario_data, full_model, process_details, seed=42)
    result = simulator.run_scenario(replications=num_reps)
    return result.get("total_cost", 0.0), result.get("avg_duration", 0.0), result.get("completed", 0)

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    # --- CONFIGURATION ---
    TARGET_COST = 45000.0
    TARGET_DURATION = 70000.0
    TARGET_WAITING_TIME = 100_000.0
    
    BASE_FILE = "data/BIMP/model/scenario.json"
    MODEL_FILE = "data/BIMP/model/model.json"
    DATA_FILE = "data/BIMP/sim_data_waiting_times.csv" 
    
    print(f"--- TARGETS ---")
    print(f"Goal Cost:     ${TARGET_COST:.2f}")
    print(f"Goal Duration: {TARGET_DURATION:.1f} seconds")
    print(f"Goal Waiting Time: {TARGET_WAITING_TIME:.1f} seconds\n")

    # STEP 0: LOAD FILES & PREP
    with open(BASE_FILE, 'r') as f: base_json = json.load(f)
    with open(MODEL_FILE, 'r') as f: full_model = json.load(f)
    
    baseline_scenario = base_json["0"]
    
    process_details = {}
    for p_id, p_data in full_model.get("process_elements", {}).items():
        if p_data.get("node_details"): 
            process_details = p_data
            break
            
    valid_node_ids = set(process_details["node_details"].keys())
    for node_id, node in process_details["node_details"].items():
        if "previous" in node and node["previous"]:
            node["previous"] = [p for p in node["previous"] if p in valid_node_ids]

    # STEP 1: EVALUATE BASELINE SCENARIO
    print("[1/4] Running Ground-Truth SimPy Evaluation on BASELINE...")
    base_true_cost, base_true_duration, base_true_completed = evaluate_scenario(
        baseline_scenario, full_model, process_details
    )

    # STEP 2: NEURAL NETWORK OPTIMIZATION
    print("\n[2/4] Running Targeted Neural Network Optimizer...")
    x_scaler = joblib.load('simple_model/output/x_scaler.pkl')
    y_scaler = joblib.load('simple_model/output/y_scaler.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = PharmacySurrogate(x_scaler.n_features_in_).to(device)
    model.load_state_dict(torch.load('simple_model/output/surrogate_model.pth', map_location=device, weights_only=True))
    model.eval()
    for param in model.parameters(): param.requires_grad = False

    df = pd.read_csv(DATA_FILE)
    
    # Feature Alignment
    X_df = df.drop(columns=["scenario_id", "kpi_total_cost", "kpi_std_total_cost", "kpi_cycle_time", "kpi_std_cycle_time", "kpi_waiting_time", "kpi_std_waiting_time"])
    X_cols = X_df.columns.tolist()
    
    raw_min_array = X_df.min().values.reshape(1, -1)
    raw_max_array = X_df.max().values.reshape(1, -1)
    
    min_scaled_bounds = torch.tensor(x_scaler.transform(raw_min_array), dtype=torch.float32, device=device)
    max_scaled_bounds = torch.tensor(x_scaler.transform(raw_max_array), dtype=torch.float32, device=device)
    
    # --- MULTI-START INITIALIZATION ---
    NUM_STARTS = 500
    
    # Spawn 499 random starts all over the parameter space
    rand_starts = min_scaled_bounds + torch.rand((NUM_STARTS - 1, len(X_cols)), device=device) * (max_scaled_bounds - min_scaled_bounds)
    
    # Add baseline as the 500th start
    base_x_raw = X_df.iloc[0].values.reshape(1, -1)
    base_tensor = torch.tensor(x_scaler.transform(base_x_raw), dtype=torch.float32, device=device)
    
    # Combine into a massive [500, 20] tensor
    batch_x_init = torch.cat([rand_starts, base_tensor], dim=0)
    x_optim = nn.Parameter(batch_x_init, requires_grad=True)

    res_amount_indices = [i for i, col in enumerate(X_cols) if col.startswith("res_") and col.endswith("_amount")]
    x_mean_tensor = torch.tensor(x_scaler.mean_, dtype=torch.float32, device=device)
    x_scale_tensor = torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device)

    y_mean_tensor = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
    y_scale_tensor = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)

    optimizer = optim.Adam([x_optim], lr=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-4)
    best_global_loss = float('inf')
    best_x_optimal = None

    for epoch in range(10000):
        optimizer.zero_grad()
        
        # Predictions shape is now [500, 3]
        predictions = model(x_optim) 
        
        # INVERSE TRANSFORM
        raw_preds = (predictions * y_scale_tensor) + y_mean_tensor
        
        # Extract the 3 raw metrics
        pred_total_cost = raw_preds[:, 0]
        pred_completed  = raw_preds[:, 1]
        pred_avg_dur    = raw_preds[:, 2]
        
        # ==========================================
        # RECONSTRUCT THE TARGET METRIC OUTSIDE THE NN
        # Add a tiny epsilon (1e-6) to prevent division by zero crashes 
        # if the NN accidentally predicts 0 completed instances.
        # ==========================================
        pred_avg_cost = pred_total_cost / (pred_completed + 1e-6)
        
        # KPI LOSS
        loss_avg_cost = ((pred_avg_cost - TARGET_COST) / TARGET_COST) ** 2
        loss_avg_dur  = ((pred_avg_dur - TARGET_DURATION) / TARGET_DURATION) ** 2
        kpi_loss = loss_avg_cost + loss_avg_dur
        
        # GUARDRAILS
        x_raw_differentiable = (x_optim * x_scale_tensor) + x_mean_tensor
        res_amounts = x_raw_differentiable[:, res_amount_indices]
        fractional_penalty = torch.sum(torch.sin(np.pi * res_amounts) ** 2, dim=1)
        
        # ANNEALING (Increase the integer weight significantly)
        integer_weight = max(0.0, (epoch - 5000) / 5000.0) * 1000.0 # Increased to 1000.0
        
        loss = (10000.0 * kpi_loss) + (integer_weight * fractional_penalty)
        min_loss_in_batch, min_idx = torch.min(loss), torch.argmin(loss)
        if min_loss_in_batch.item() < best_global_loss:
            best_global_loss = min_loss_in_batch.item()
            best_x_optimal = x_optim[min_idx].detach().clone()
        batch_loss = loss.sum()
        batch_loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            x_optim.clamp_(min_scaled_bounds, max_scaled_bounds)

    # --- FIND THE BEST RESULT ---
    # Out of the 500 paths, find the single index that reached the lowest loss
    best_idx = torch.argmin(loss).item()
    
    optimized_x_raw = x_scaler.inverse_transform(best_x_optimal.cpu().numpy().reshape(1, -1))[0]

    # STEP 3: INJECT OPTIMIZED PARAMETERS WITH PHYSICAL CONSTRAINTS
    print("[3/4] Injecting Optimized Parameters and Formatting...")
    opt_scenario = copy.deepcopy(baseline_scenario)
    discretized_x_raw = np.copy(optimized_x_raw)
    
    for i, col in enumerate(X_cols):
        val = optimized_x_raw[i]
        
        if col == "arrival_rate_mean":
            val = max(120.0, round(val, 2))
            opt_scenario["arrivalRateDistribution"]["mean"] = str(val)
            
        elif col.startswith("res_") and col.endswith("_amount"):
            res_name = col.replace("res_", "").replace("_amount", "").replace("_", " ")
            val = max(1, min(15, int(round(val)))) 
            for r in opt_scenario.get("resources", []):
                if r["name"] == res_name:
                    r["totalAmount"] = str(val)
                    
        elif col.startswith("res_") and col.endswith("_cost"):
            res_name = col.replace("res_", "").replace("_cost", "").replace("_", " ")
            val = max(10.00, round(val, 2))
            for r in opt_scenario.get("resources", []):
                if r["name"] == res_name:
                    r["costPerHour"] = str(val)
                    
        elif col.startswith("el_") and col.endswith("_duration"):
            el_id = col.replace("el_", "").replace("_duration", "")
            val = max(1.0, round(val, 2))
            for el in opt_scenario.get("elements", []):
                if el["elementId"] == el_id:
                    orig_mean = float(el["durationDistribution"]["mean"])
                    orig_std = float(el.get("durationDistribution", {}).get("standardDeviation", 0))
                    
                    el["durationDistribution"]["mean"] = str(val)
                    
                    # Keep the CV scaling intact so the SimPy simulator doesn't crash, 
                    # even though we aren't optimizing for it anymore.
                    if orig_mean > 0 and orig_std > 0:
                        cv = orig_std / orig_mean
                        new_std = round(val * cv, 2)
                        el["durationDistribution"]["standardDeviation"] = str(new_std)
        
        discretized_x_raw[i] = val

    discretized_x_scaled = x_scaler.transform(discretized_x_raw.reshape(1, -1))
    discretized_tensor = torch.tensor(discretized_x_scaled, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        final_pred_scaled = model(discretized_tensor).cpu().numpy()

    final_pred = y_scaler.inverse_transform(final_pred_scaled)[0]

    # STEP 4: EVALUATE OPTIMIZED SCENARIO
    print("[4/4] Running Ground-Truth SimPy Evaluation on OPTIMIZED...")
    opt_true_cost, opt_true_duration, opt_true_completed = evaluate_scenario(
        opt_scenario, full_model, process_details
    )

    # RESULTS REPORTING - STRIPPED OF ALL STD DEVIATION
    print("\n=====================================================================")
    print("                    VALIDATION & ROI REPORT")
    print("=====================================================================")
    print(f"                | COST (Average)       | DURATION (Average)")
    print("---------------------------------------------------------------------")
    print(f"TARGET GOAL     | ${TARGET_COST:<19.2f} | {TARGET_DURATION:.1f}s")
    print(f"BASELINE (True) | ${(base_true_cost/base_true_completed):<19.2f} | {base_true_duration:.1f}s")
    print(f"NN PREDICTED    | ${final_pred[0]/final_pred[1]:<19.2f} | {(final_pred[2]):.1f}s")
    print(f"OPTIMIZED (True)| ${(opt_true_cost/opt_true_completed):<19.2f} | {opt_true_duration:.1f}s")
    print("---------------------------------------------------------------------")
    
    start_cost_diff = round(base_true_cost/base_true_completed, 2) - TARGET_COST
    start_dur_diff = base_true_duration - TARGET_DURATION
    end_cost_diff = round(opt_true_cost/opt_true_completed, 2) - TARGET_COST
    end_dur_diff = opt_true_duration - TARGET_DURATION
    
    print(f"STARTING DELTA      | {('+' if start_cost_diff > 0 else '')}${start_cost_diff:<19.2f} | {('+' if start_dur_diff > 0 else '')}{start_dur_diff:.1f}s")
    print("=====================================================================")

    print(f"FINISHING DELTA      | {('+' if end_cost_diff > 0 else '')}${end_cost_diff:<19.2f} | {('+' if end_dur_diff > 0 else '')}{end_dur_diff:.1f}s")
    print("=====================================================================")

if __name__ == "__main__":
    main()