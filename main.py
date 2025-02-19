import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('data/BFData_02.xlsx')
# First 13 seconds of signal is very messy (can remove for other files)
df = df[df['Time (s)'] >= 13]
# Convert all columns to numeric, forcing non-convertible values to NaN
df = df.apply(pd.to_numeric, errors='coerce')

#Butterworth filter - sampling rate = 100Hz, 2nd order, higher cutoff
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=1.0, fs=1000, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff/nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

for col in df.columns:
    if col not in ('Time (s)', 'HR'):
        df[col] = butter_lowpass_filter(df[col])

print(df.describe())