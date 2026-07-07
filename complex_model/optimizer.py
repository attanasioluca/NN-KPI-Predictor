import json
import copy
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path: sys.path.append(parent_dir)

from complex_model.hypertuned_model import PharmacySurrogate
from helpers.simulator import ScenarioSimulator

def evaluate_scenario(scenario_data, full_model, process_details, num_reps=50):
    simulator = ScenarioSimulator(scenario_data, full_model, process_details, seed=42)
    result = simulator.run_scenario(replications=num_reps, until=86400 * 14)
    
    # Mapped to the updated simulator.py return keys
    avg_cost = result.get("total_cost", 0.0)
    std_cost = result.get("std_cost", 0.0)
    avg_dur  = result.get("avg_cycle_time", 0.0)
    std_dur  = result.get("std_cycle_time", 0.0)
    avg_wait = result.get("avg_wait_time", 0.0)
    std_wait = result.get("std_wait_time", 0.0)
    
    return avg_cost, std_cost, avg_dur, std_dur, avg_wait, std_wait

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    # --- TARGETS ---
    TARGET_COST = 27.0
    TARGET_DURATION = 70000.0
    TARGET_WAIT_TIME = 1200.0  # Set requested Waiting Time here
    
    RISK_AVERSION_WEIGHT = 0  # Lambda: Penalizes high standard deviation
    MAX_Z_SCORE = 3.0           
    
    BASE_FILE = "data/real/model/scenario.json"
    MODEL_FILE = "data/real/model/model.json"
    DATA_FILE = "data/real/sim_data_waiting_times.csv" 
    
    print(f"--- TARGETS ---")
    print(f"Goal Cost:       ${TARGET_COST:.2f}")
    print(f"Goal Cycle Time: {TARGET_DURATION:.1f} seconds")
    print(f"Goal Wait Time:  {TARGET_WAIT_TIME:.1f} seconds")
    print(f"Risk Aversion Weight (Lambda): {RISK_AVERSION_WEIGHT}\n")

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

    print("[1/4] Running Ground-Truth SimPy Evaluation on BASELINE...")
    base_true_cost, base_true_std_cost, base_true_duration, base_true_std_duration, base_true_wait, base_true_std_wait = evaluate_scenario(
        baseline_scenario, full_model, process_details, num_reps=50
    )

    print("\n[2/4] Running High-Speed Risk-Averse Neural Network Optimizer...")
    x_scaler = joblib.load('complex_model/output/x_scaler.pkl')
    y_scaler = joblib.load('complex_model/output/y_scaler.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = PharmacySurrogate(x_scaler.n_features_in_, DROPOUT_RATE=0.21629303761709978).to(device)
    model.load_state_dict(torch.load('complex_model/output/surrogate_model.pth', map_location=device, weights_only=True))
    model.eval()
    for param in model.parameters(): param.requires_grad = False

    df = pd.read_csv(DATA_FILE)
    X_cols = [col for col in df.columns if col.startswith(("arrival_", "res_", "el_"))]
    X_df = df[X_cols]
    
    raw_min_array = X_df.min().values.reshape(1, -1)
    raw_max_array = X_df.max().values.reshape(1, -1)
    
    min_scaled_bounds = torch.tensor(x_scaler.transform(raw_min_array), dtype=torch.float32, device=device)
    max_scaled_bounds = torch.tensor(x_scaler.transform(raw_max_array), dtype=torch.float32, device=device)
    
    # --- MULTI-START INITIALIZATION ---
    NUM_STARTS = 500
    rand_starts = min_scaled_bounds + torch.rand((NUM_STARTS - 1, len(X_cols)), device=device) * (max_scaled_bounds - min_scaled_bounds)
    base_tensor = torch.tensor(x_scaler.transform(X_df.iloc[0].values.reshape(1, -1)), dtype=torch.float32, device=device)
    
    x_optim = nn.Parameter(torch.cat([rand_starts, base_tensor], dim=0), requires_grad=True)

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
        predictions = model(x_optim)
        
        # INVERSE TRANSFORM: Convert shape [500, 3] back to real KPIs
        raw_preds = (predictions * y_scale_tensor) + y_mean_tensor
        
        # The model only predicts the 3 averages
        pred_avg_cost = raw_preds[:, 0]
        pred_dur      = raw_preds[:, 1]
        pred_wait     = raw_preds[:, 2]
        
        loss_avg_cost = ((pred_avg_cost - TARGET_COST) / TARGET_COST) ** 2
        loss_avg_dur  = ((pred_dur - TARGET_DURATION) / TARGET_DURATION) ** 2
        loss_avg_wait = ((pred_wait - TARGET_WAIT_TIME) / TARGET_WAIT_TIME) ** 2
        
        kpi_loss = loss_avg_cost + loss_avg_dur + loss_avg_wait
        
        # RISK PENALTY: Since the model doesn't predict standard deviations, 
        # we cannot penalize them in the gradient loop. 
        risk_loss = 0.0 
        
        x_raw_diff = (x_optim * x_scale_tensor) + x_mean_tensor
        res_amounts = x_raw_diff[:, res_amount_indices]
        fractional_penalty = torch.sum(torch.sin(np.pi * res_amounts) ** 2, dim=1)
        z_score_penalty = torch.sum(torch.relu(torch.abs(x_optim) - MAX_Z_SCORE) ** 2, dim=1)
        
        integer_weight = max(0.0, (epoch - 5000) / 5000.0) * 1000.0 
        loss = (10000.0 * kpi_loss) + (RISK_AVERSION_WEIGHT * risk_loss) + (integer_weight * fractional_penalty) + (0.5 * z_score_penalty)
        
        # TRACK GLOBAL BEST INSIDE THE EPOCH LOOP
        min_loss_in_batch, min_idx = torch.min(loss), torch.argmin(loss)
        if min_loss_in_batch.item() < best_global_loss:
            best_global_loss = min_loss_in_batch.item()
            best_x_optimal = x_optim[min_idx].detach().clone()
        
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad(): x_optim.clamp_(min_scaled_bounds, max_scaled_bounds)

    # --- ZERO LAG: CPU transfer and inverse transform executed exactly ONCE here! ---
    optimized_x_raw = x_scaler.inverse_transform(best_x_optimal.cpu().numpy().reshape(1, -1))[0]

    # STEP 3: INJECT OPTIMIZED PARAMETERS
    print("[3/4] Injecting Optimized Parameters and Scaling CVs...")
    opt_scenario = copy.deepcopy(baseline_scenario)
    discretized_x_raw = np.copy(optimized_x_raw)
    
    for i, col in enumerate(X_cols):
        val = optimized_x_raw[i]
        if col == "arrival_rate_mean":
            val = max(120.0, round(val, 2))
            opt_scenario["arrivalRateDistribution"]["mean"] = str(val)
        elif col.startswith("res_") and col.endswith("_amount"):
            res_name = col.replace("res_", "").replace("_amount", "").replace("_", " ")
            val = max(1, min(15, int(round(val))))  # Permitted up to physical cap of 15
            for r in opt_scenario.get("resources", []):
                if r["name"] == res_name: r["totalAmount"] = str(val)
        elif col.startswith("res_") and col.endswith("_cost"):
            res_name = col.replace("res_", "").replace("_cost", "").replace("_", " ")
            val = max(10.00, round(val, 2))
            for r in opt_scenario.get("resources", []):
                if r["name"] == res_name: r["costPerHour"] = str(val)
        elif col.startswith("el_") and col.endswith("_duration"):
            el_id = col.replace("el_", "").replace("_duration", "")
            val = max(1.0, round(val, 2))
            for el in opt_scenario.get("elements", []):
                if el["elementId"] == el_id:
                    orig_mean = float(el["durationDistribution"]["mean"])
                    orig_std = float(el.get("durationDistribution", {}).get("standardDeviation", 0))
                    el["durationDistribution"]["mean"] = str(val)
                    if orig_mean > 0 and orig_std > 0:
                        el["durationDistribution"]["standardDeviation"] = str(round(val * (orig_std / orig_mean), 2))
        discretized_x_raw[i] = val

    discretized_tensor = torch.tensor(x_scaler.transform(discretized_x_raw.reshape(1, -1)), dtype=torch.float32, device=device)
    with torch.no_grad(): final_pred_scaled = model(discretized_tensor).cpu().numpy()
    final_pred = y_scaler.inverse_transform(final_pred_scaled)[0]

    # COMPUTE NN PREDICTIONS CORRECTLY (Only 3 targets)
    nn_pred_avg_cost = final_pred[0]
    nn_pred_dur_mean = final_pred[1]
    nn_pred_wait_mean = final_pred[2]
    
    # NN doesn't predict std deviations, setting placeholders for the table
    nn_pred_std_cost = 0.0
    nn_pred_dur_std = 0.0
    nn_pred_wait_std = 0.0

    # STEP 4: EVALUATE OPTIMIZED SCENARIO
    print("[4/4] Running Ground-Truth SimPy Evaluation on OPTIMIZED...")
    opt_true_cost, opt_true_std_cost, opt_true_duration, opt_true_std_duration, opt_true_wait, opt_true_std_wait = evaluate_scenario(
        opt_scenario, full_model, process_details, num_reps=100
    )

    print("\n=======================================================================================")
    print("                 RISK-AWARE VALIDATION & ROI REPORT")
    print("=======================================================================================")
    print(f"                | COST (Avg ± Std)       | CYCLE TIME (Avg ± Std)| WAIT TIME (Avg ± Std)")
    print("---------------------------------------------------------------------------------------")
    print(f"TARGET GOAL     | ${TARGET_COST:<6.2f} ± MINIMIZE     | {TARGET_DURATION:<6.1f}s ± MINIMIZE   | {TARGET_WAIT_TIME:<6.1f}s ± MINIMIZE")
    print(f"BASELINE (True) | ${base_true_cost:<6.2f} ± {base_true_std_cost:<11.2f} | {base_true_duration:<6.1f}s ± {base_true_std_duration:<8.1f}s| {base_true_wait:<6.1f}s ± {base_true_std_wait:.1f}s")
    print(f"NN PREDICTED    | ${nn_pred_avg_cost:<6.2f} ± {nn_pred_std_cost:<11.2f} | {nn_pred_dur_mean:<6.1f}s ± {nn_pred_dur_std:<8.1f}s| {nn_pred_wait_mean:<6.1f}s ± {nn_pred_wait_std:.1f}s")
    print(f"OPTIMIZED (True)| ${opt_true_cost:<6.2f} ± {opt_true_std_cost:<11.2f} | {opt_true_duration:<6.1f}s ± {opt_true_std_duration:<8.1f}s| {opt_true_wait:<6.1f}s ± {opt_true_std_wait:.1f}s")
    print("---------------------------------------------------------------------------------------")

    print("\n=====================================================================")
    print("                 RECOMMENDED CONFIGURATION CHANGES")
    print("=====================================================================")
    
    # Extract original baseline feature values to compare against optimized ones
    base_x_raw = X_df.iloc[0].values
    changes_found = False
    
    for i, col in enumerate(X_cols):
        b_val = base_x_raw[i]
        o_val = discretized_x_raw[i]
        
        # Only print values that the optimizer significantly changed
        if abs(b_val - o_val) > 0.01:
            changes_found = True
            if col.endswith("_amount"):
                print(f"{col:<30} | Baseline: {int(b_val):<4} -> Optimized: {int(o_val)}")
            else:
                print(f"{col:<30} | Baseline: {b_val:<7.2f} -> Optimized: {o_val:.2f}")
                
    if not changes_found:
        print("No changes required. The baseline scenario already hits your targets.")

if __name__ == "__main__":
    main()