import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import time

# ==========================================
# LINEAR REGRESSION BASELINE
# ==========================================
def main():
    DATA_FILE = "data/BIMP/sim_data_waiting_times.csv"
    
    print("Loading dataset...")
    # 1. Load the ENTIRE dataset (remove nrows)
    df = pd.read_csv(DATA_FILE) 
    
    X_df = df.drop(columns=["scenario_id", "kpi_total_cost", "kpi_std_total_cost", "kpi_cycle_time", "kpi_std_cycle_time", "kpi_waiting_time", "kpi_std_waiting_time", "n_reps_used", "converged","converged_wait","converged_cost","converged_duration"])
    y_df = df[["kpi_total_cost", "kpi_cycle_time", "kpi_waiting_time" ]]
    
    input_size = X_df.shape[1]
    output_size = y_df.shape[1]
    print(f"Features: {input_size} | Targets: {output_size}")

    # 2. Split the FULL dataset once to create a universal, locked test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_df.values, y_df.values, test_size=0.2, random_state=42
    )

    # 3. Define how many training samples you want to use for this specific run
    # CHANGE THIS VARIABLE to fill your table (1000, 2500, 3500, 7500, etc.)
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
    
    # Save scalers specifically for the LR model (so it doesn't overwrite your NN ones)
    joblib.dump(x_scaler, 'simple_model/output/lr_x_scaler.pkl')
    joblib.dump(y_scaler, 'simple_model/output/lr_y_scaler.pkl')

    print("\nTraining Linear Regression Model...")
    start_time = time.time()
    
    # Scikit-learn's LinearRegression handles multiple outputs natively
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Save the model
    joblib.dump(model, 'simple_model/output/lr_model.pkl')
    
    train_time = time.time() - start_time
    print(f"Training complete in {train_time:.4f} seconds!")

    # ==========================================
    # EVALUATION & METRICS
    # ==========================================
    print("\nEvaluating on Test Set...")
    
    # Predict in scaled space
    predictions_scaled = model.predict(X_test_scaled)
    
    # 1. Scaled Metrics
    test_mse_scaled = np.mean((predictions_scaled - y_test_scaled) ** 2, axis=0)
    test_rmse_scaled = np.sqrt(test_mse_scaled)
    
    # 2. Apply your formulas
    error_kpi = test_rmse_scaled * y_scaler.scale_                  # error(kpi) = RMSE(kpi) * std_dev(kpi)
    percentage_kpi = (error_kpi / np.abs(y_scaler.mean_)) * 100     # percentage(kpi) = error / avg * 100
    mse_kpi = error_kpi ** 2                                        # MSE in real terms
    
    print("\n=====================================================================")
    print("                    LINEAR REGRESSION RESULTS")
    print("=====================================================================")
    print(f"Test Error -> Cost:      ${error_kpi[0]:.2f} (±{percentage_kpi[0]:.1f}%)")
    print(f"Test Error -> Cycle Time: {error_kpi[1]:.2f} (±{percentage_kpi[1]:.1f}%)")
    print(f"Test Error -> Wait Time:  {error_kpi[2]:.1f}s (±{percentage_kpi[2]:.1f}%)")
    print("=====================================================================")

    # ==========================================
    # JSON EXPORT
    # ==========================================
    import json
    metrics = {
        "model_name": "Linear Regression (Baseline)",
        "Best Test Loss":float(np.mean(test_mse_scaled)),  #What's here ??,
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
    
    with open("output/metrics_lr_model.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()