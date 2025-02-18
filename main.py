import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('data/BFData_02.xlsx')
df = df[df['Time (s)'] >= 13]
# Convert all columns to numeric, forcing non-convertible values to NaN
df = df.apply(pd.to_numeric, errors='coerce')

#Butterworth filter - sampling rate = 100Hz, 2nd order, higher cutoff

from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=1.0, fs=100, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff/nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

for col in df.columns:
    if col not in ('Time (s)', 'HR'):
        df[col] = butter_lowpass_filter(df[col])

import matplotlib.pyplot as plt

# Define specific time points to mark
labchart_times = [1062, 1122, 1185, 1249, 1263]  # Replace with your actual LabChart timestamps
adjusted_conditions = [t - 897 for t in labchart_times]

def find_nearest_time(labchart_time, time_offset=897):
    """Convert LabChart time to adjusted time and find the nearest index in the dataset."""
    adjusted_time = labchart_time - time_offset
    nearest_index = (df["Time (s)"] - adjusted_time).abs().idxmin()
    return df.loc[nearest_index, "Time (s)"]  # Return the closest available time


# Define adjusted condition times (from previous step)
condition_times = adjusted_conditions  # Replace with real values

# Plot HR over time
plt.figure(figsize=(10, 5))
plt.plot(df["Time (s)"], df["MCAv_mean"], label="MCAv_mean", color="blue")

# Add vertical lines for condition changes
for t in condition_times:
    plt.axvline(x=t, color='green', linestyle='--', label=f"Condition Change at {t:.1f}s")
    plt.text(t, df["MCAv_mean"].max(), f"{t:.1f}s", rotation=90, verticalalignment='top', fontsize=12)

# Add labels and legend
plt.xlabel("Time (s)")
plt.ylabel("Mean MCV")
plt.title("Mean MCV over Time")
plt.legend()
plt.grid(True)
plt.show()

