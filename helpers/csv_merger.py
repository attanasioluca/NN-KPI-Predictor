import pandas as pd

df1 = pd.read_csv("data/real/mac_sim_data_waiting_times.csv")
df2 = pd.read_csv("data/real/pc_sim_data_waiting_times.csv")

df = pd.concat([df1, df2], ignore_index=True)

# Sort by scenarioId
df = df.sort_values(by="scenario_id").reset_index(drop=True)

df.to_csv("data/real/sim_data_waiting_times.csv", index=False)

print(len(df))