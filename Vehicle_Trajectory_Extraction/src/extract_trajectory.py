import pandas as pd
import numpy as np

# Load trajectory data
csv_path = "output/trajectory.csv"
df = pd.read_csv(csv_path)

# Compute additional motion parameters
df['Acceleration'] = df['Speed [km/h]'].diff() / df['Time [s]'].diff()
df['Acceleration'].fillna(0, inplace=True)

df.to_csv("output/trajectory_enriched.csv", index=False)
print("Trajectory extraction completed and saved.")
