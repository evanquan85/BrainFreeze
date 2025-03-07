import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

normoxic_file_path = 'data/BFData_01.csv' # Replace with actual file path

def load_data(normoxic_file_path):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(normoxic_file_path)


def remove_spikes(df, factors, window_size):
    """Apply a rolling median filter with a larger window to remove more aggressive spikes."""
    for factor in factors:
        # Convert column to numeric, setting errors='coerce' will replace non-numeric values with NaN
        df[factor] = pd.to_numeric(df[factor], errors='coerce')
        # Apply rolling median filter
        df[factor] = df[factor].rolling(window=window_size, center=True).median()
    return df


def analyze_brain_freeze(normoxic_file_path):
    """Main function to perform analysis."""
    df = load_data(normoxic_file_path)

    # Ensure copy to avoid SettingWithCopyWarning
    df = df.copy()
    # Convert columns to numeric
    df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors='coerce')
    df["MCAv_mean"] = pd.to_numeric(df["MCAv_mean"], errors='coerce')
    df["MCAv_raw"] = pd.to_numeric(df["MCAv_raw"], errors='coerce')
    # Define time-based conditions for baseline and brain freeze
    baseline_df = df[df["Time (s)"] <= 60] # First 60 seconds are baseline
    brain_freeze_df = df[df["Time (s)"] >= 189] # Brain freeze starts at 185-195 seconds (EDIT THIS FOR EACH FILE)

    # Remove HR spike between 239-241 seconds and interpolate the values - ONLY IF OUTLIER VALUE
    hr_spike_mask = (brain_freeze_df["Time (s)"] >= 239) & (brain_freeze_df["Time (s)"] <= 241)
    brain_freeze_df.loc[hr_spike_mask, "HR"] = np.nan
    brain_freeze_df = brain_freeze_df.copy()  # Ensure it's a copy
    brain_freeze_df.loc[:, "HR"] = brain_freeze_df["HR"].interpolate(method='linear')

    # Extracting relevant columns for analysis (excluding Time and unnamed columns)
    factors = ['MCAv_mean', 'MCAv_dia', 'MCAv_raw', 'MCAv_sys', 'FP_raw',
               'HCU_pressure', 'Systolic', 'Mean_arterial', 'Diastolic', 'HR', 'MCA_PI']

    # Apply spike removal filter with a more aggressive window size
    df = remove_spikes(df, factors, window_size=5)

    # Ensure equal sample sizes for comparison AFTER spike removal
    min_length = min(len(baseline_df), len(brain_freeze_df))
    baseline_sample = baseline_df.iloc[:min_length].reset_index(drop=True)
    brain_freeze_sample = brain_freeze_df.iloc[:min_length].reset_index(drop=True)

    # Normalize time to be relative within each condition
    baseline_sample["Relative Time (s)"] = baseline_sample["Time (s)"] - baseline_sample["Time (s)"].iloc[0]
    brain_freeze_sample["Relative Time (s)"] = brain_freeze_sample["Time (s)"] - brain_freeze_sample["Time (s)"].iloc[0]

    # Calculate peak values and time to peak for each factor
    results = []
    for factor in factors:
        baseline_peak = round(baseline_sample[factor].max(), 4)
        baseline_time = round(baseline_sample.loc[baseline_sample[factor].idxmax(), "Relative Time (s)"], 4)

        brain_freeze_peak = round(brain_freeze_sample[factor].max(), 4)
        brain_freeze_time = round(brain_freeze_sample.loc[brain_freeze_sample[factor].idxmax(), "Relative Time (s)"], 4)

        percent_change = round(((brain_freeze_peak - baseline_peak) / baseline_peak) * 100, 4)

        # Perform dependent two-tailed t-test, handling NaNs and ensuring equal-length samples
        clean_baseline = baseline_sample[factor].dropna().values
        clean_brain_freeze = brain_freeze_sample[factor].dropna().values

        # Trim longer array to match the shorter one
        min_len = min(len(clean_baseline), len(clean_brain_freeze))
        if min_len > 1:
            t_stat, p_value = stats.ttest_rel(clean_baseline[:min_len], clean_brain_freeze[:min_len])
            t_stat = round(t_stat, 4)
            p_value = round(p_value, 4)
        else:
            t_stat, p_value = np.nan, np.nan  # Assign NaN if test cannot be performed

        results.append({
            "Factor": factor,
            "Baseline Peak": baseline_peak,
            "Baseline Time to Peak": baseline_time,
            "Brain Freeze Peak": brain_freeze_peak,
            "Brain Freeze Time to Peak": brain_freeze_time,
            "% Change": percent_change,
            "T-Statistic": t_stat,
            "P-Value": p_value
        })

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  # Set output width to avoid truncation
    print(results_df)
    # Save dataframe print output as CSV - MODIFY FOR EACH PARTICIPANT
    results_df.to_csv('PrintOutputs/Brain_Freeze_Results002.csv', index=False)

    # Plot comparison for each factor as overlapped line Graphs with relative time
    '''
    plt.figure(figsize=(12, 6))
    for factor in factors:
        plt.figure()
        plt.plot(baseline_sample["Relative Time (s)"], baseline_sample[factor], label="Baseline", color='blue')
        plt.plot(brain_freeze_sample["Relative Time (s)"], brain_freeze_sample[factor], label="Brain Freeze",
                 color='red')
        plt.xlabel("Relative Time (s)")
        plt.ylabel(factor)
        plt.title(f"Comparison of {factor} Over Relative Time")
        plt.legend()
        plt.show()
    '''
    # Plot the data with MCAv_mean as a variable trace instead of a polynomial trendline
    plt.figure(figsize=(12, 6))

    # Plot MCAv_raw as background in gray
    plt.plot(df["Time (s)"], df["MCAv_raw"], label="MCAv_raw", color='gray', alpha=0.5)

    # Plot MCAv_mean as a more variable trace in red
    plt.plot(df["Time (s)"], df["MCAv_mean"], label="MCAv_mean", color='red', linewidth=2)

    # Add a vertical dashed line at 189 seconds
    plt.axvline(x=189, color='blue', linestyle='dashed', label="Brain Freeze Start")
    plt.axvline(x=204, color='green', linestyle='dashed', label="Brain Freeze Achieved")
    plt.axvline(x=60, color='black', linestyle='dashed', label='Resting')

    # Add text annotation for "Resting" slightly more to the left
    plt.text(20, df["MCAv_raw"].min(), "Resting", color='black', fontsize=12, verticalalignment='bottom')

    # Labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("MCA Velocity")
    plt.title("Mean MCA & Raw MCA Velocity Over Time")
    plt.legend()

    # Save plot with higher DPI (e.g., 300 for print-quality) - RENAME EACH FOR EACH PARTICIPANT
    plt.savefig("Graphs/MCA_Velocity_Comparison002.png", dpi=600, bbox_inches="tight")

    # Show the plot
    plt.show()

# REMEMBER TO CALL FUNCTION
analyze_brain_freeze(normoxic_file_path)
