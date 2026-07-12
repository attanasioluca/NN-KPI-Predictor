import pandas as pd

df1 = pd.read_csv("data/synthetic/sim_data_waiting_times.csv")
df2 = pd.read_csv("data/synthetic/3_sim_data_waiting_times.csv")

df = pd.concat([df1, df2], ignore_index=True)

# Drop unwanted columns (ignore errors in case one is missing)
df = df.drop(
    columns=[
        "kpi_std_total_cost",
        "kpi_std_cycle_time",
        "kpi_std_waiting_time",
    ],
    errors="ignore",
)

# Sort by scenario_id
df = df.sort_values(by="scenario_id").reset_index(drop=True)

df.to_csv("data/synthetic/sim_data_waiting_times.csv", index=False)

print(len(df))