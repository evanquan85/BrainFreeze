import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    'Baseline': pd.read_excel('data/BFAnalysisData_01.xlsx', sheet_name='c. 7-8 (BL - TNW)'),
    'Normal Water': pd.read_excel('data/BFAnalysisData_01.xlsx', sheet_name='c. 8-10 (TNW - IW)'),
    'Ice Water': pd.read_excel('data/BFAnalysisData_01.xlsx', sheet_name='c. 10-12 (IW - BFP)'),
    'Brain Freeze': pd.read_excel('data/BFAnalysisData_01.xlsx', sheet_name='c. 12-13 (BFP - BF Ach.)'),
    'Recovery': pd.read_excel('data/BFAnalysisData_01.xlsx', sheet_name='c. 13-14 (BF ACh. - End)')
}

plt.figure(figsize=(12, 8))
for i, (label, df) in enumerate(data.items(), 1):
    df['MCAv_mean'] = pd.to_numeric(df['MCAv_mean'], errors='coerce')
    df = df.dropna(subset=['MCAv_mean'])

    plt.subplot(3, 2, i)
    plt.plot(df['MCAv_mean'], label=label)
    plt.title(label)
    plt.xlabel('Time')
    plt.ylabel('MCAv_mean')
    plt.legend()

plt.tight_layout()
plt.show()

