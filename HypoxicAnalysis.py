import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

hypoxic_file_path = 'data/'  # Replace with actual file path

def load_data(file_path):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def remove_spikes(df, factors, window_size=250):
    """Apply a rolling median filter with a larger window to remove more aggressive spikes."""
    for factor in factors:
        df[factor] = df[factor].rolling(window=window_size, center=True).median()
    return df


def analyze_hypoxia(file_path):
    """Main function to perform hypoxic analysis."""
    df = load_data(file_path)

    # Define time-based conditions for baseline and hypoxia
    baseline_df = df[df["Time (s)"] <= 60]  # First 60 seconds are baseline
    hypoxia_start_time = 200  # Adjust this based on the dataset
    hypoxia_df = df[df["Time (s)"] >= hypoxia_start_time]

    # Ensure copy to avoid SettingWithCopyWarning
    hypoxia_df = hypoxia_df.copy()

    # Extracting relevant columns for analysis (excluding Time and unnamed columns)
    factors = ['MCAv_mean', 'MCAv_dia', 'MCAv_raw', 'MCAv_sys', 'FP_raw',
               'HCU_pressure', 'Systolic', 'Mean_arterial', 'Diastolic', 'HR', 'MCA_PI']

    # Apply spike removal filter with a more aggressive window size
    df = remove_spikes(df, factors, window_size=250)
    baseline_df = df[df["Time (s)"] <= 60]
    hypoxia_df = df[df["Time (s)"] >= hypoxia_start_time]

    # Ensure equal sample sizes for comparison AFTER spike removal
    min_length = min(len(baseline_df), len(hypoxia_df))
    baseline_sample = baseline_df.iloc[:min_length].reset_index(drop=True)
    hypoxia_sample = hypoxia_df.iloc[:min_length].reset_index(drop=True)

    # Normalize time to be relative within each condition
    baseline_sample["Relative Time (s)"] = baseline_sample["Time (s)"] - baseline_sample["Time (s)"].iloc[0]
    hypoxia_sample["Relative Time (s)"] = hypoxia_sample["Time (s)"] - hypoxia_sample["Time (s)"].iloc[0]

    # Calculate peak values and time to peak for each factor
    results = []
    for factor in factors:
        baseline_peak = round(baseline_sample[factor].max(), 4)
        baseline_time = round(baseline_sample.loc[baseline_sample[factor].idxmax(), "Relative Time (s)"], 4)

        hypoxia_peak = round(hypoxia_sample[factor].max(), 4)
        hypoxia_time = round(hypoxia_sample.loc[hypoxia_sample[factor].idxmax(), "Relative Time (s)"], 4)

        percent_change = round(((hypoxia_peak - baseline_peak) / baseline_peak) * 100, 4)

        # Perform dependent two-tailed t-test, handling NaNs and ensuring equal-length samples
        clean_baseline = baseline_sample[factor].dropna().values
        clean_hypoxia = hypoxia_sample[factor].dropna().values

        # Trim longer array to match the shorter one
        min_len = min(len(clean_baseline), len(clean_hypoxia))
        if min_len > 1:
            t_stat, p_value = stats.ttest_rel(clean_baseline[:min_len], clean_hypoxia[:min_len])
            t_stat = round(t_stat, 4)
            p_value = round(p_value, 4)
        else:
            t_stat, p_value = np.nan, np.nan  # Assign NaN if test cannot be performed

        results.append({
            "Factor": factor,
            "Baseline Peak": baseline_peak,
            "Baseline Time to Peak": baseline_time,
            "Hypoxia Peak": hypoxia_peak,
            "Hypoxia Time to Peak": hypoxia_time,
            "% Change": percent_change,
            "T-Statistic": t_stat,
            "P-Value": p_value
        })

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  # Set output width to avoid truncation
    print(results_df)

    # Plot comparison for each factor as overlapped line graphs with relative time
    plt.figure(figsize=(12, 6))
    for factor in factors:
        plt.figure()
        plt.plot(baseline_sample["Relative Time (s)"], baseline_sample[factor], label="Baseline", color='blue')
        plt.plot(hypoxia_sample["Relative Time (s)"], hypoxia_sample[factor], label="Hypoxia", color='red')
        plt.xlabel("Relative Time (s)")
        plt.ylabel(factor)
        plt.title(f"Comparison of {factor} Over Relative Time")
        plt.legend()
        plt.show()

#REMBEMBER TO CALL FUNCTION
analyze_hypoxia(hypoxic_file_path)
