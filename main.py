import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def analyze_brain_freeze(file_path):
    """Main function to perform analysis."""
    df = load_data(file_path)

    # Define time-based conditions for baseline and brain freeze
    baseline_df = df[df["Time (s)"] <= 60]  # First 60 seconds are baseline
    brain_freeze_df = df[df["Time (s)"] >= 187]  # Brain freeze starts at 187 seconds

    # Ensure equal sample sizes for comparison
    min_length = min(len(baseline_df), len(brain_freeze_df))
    baseline_sample = baseline_df.iloc[:min_length].reset_index(drop=True)
    brain_freeze_sample = brain_freeze_df.iloc[:min_length].reset_index(drop=True)

    # Normalize time to be relative within each condition
    baseline_sample["Relative Time (s)"] = baseline_sample["Time (s)"] - baseline_sample["Time (s)"].iloc[0]
    brain_freeze_sample["Relative Time (s)"] = brain_freeze_sample["Time (s)"] - brain_freeze_sample["Time (s)"].iloc[0]

    # Extracting relevant columns for analysis (excluding Time and unnamed columns)
    factors = ['MCAv_mean', 'MCAv_dia', 'MCAv_raw', 'MCAv_sys', 'FP_raw',
               'HCU_pressure', 'Systolic', 'Mean_arterial', 'Diastolic', 'HR', 'MCA_PI']

    # Calculate peak values and time to peak for each factor
    results = []
    for factor in factors:
        baseline_peak = round(baseline_sample[factor].max(), 4)
        baseline_time = round(baseline_sample.loc[baseline_sample[factor].idxmax(), "Time (s)"], 4)

        brain_freeze_peak = round(brain_freeze_sample[factor].max(), 4)
        brain_freeze_time = round(brain_freeze_sample.loc[brain_freeze_sample[factor].idxmax(), "Time (s)"], 4)

        percent_change = round(((brain_freeze_peak - baseline_peak) / baseline_peak) * 100, 4)

        # Perform dependent two-tailed t-test
        t_stat, p_value = stats.ttest_rel(baseline_sample[factor].values, brain_freeze_sample[factor].values)
        t_stat = round(t_stat, 4)
        p_value = round(p_value, 4)

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
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Set width to avoid line breaks
    pd.set_option('display.colheader_justify', 'center')  # Center column headers
    pd.set_option('display.float_format', '{:.4f}'.format)  # Ensure consistent decimal places
    print(results_df)


    # Plot comparison for each factor as overlapped line graphs with relative time
    plt.figure(figsize=(12, 6))
    for factor in factors:
        plt.figure()
        plt.plot(baseline_sample["Relative Time (s)"], baseline_sample[factor], label="Baseline", color='blue')
        plt.plot(brain_freeze_sample["Relative Time (s)"], brain_freeze_sample[factor], label="Brain Freeze", color='red')
        plt.xlabel("Relative Time (s)")
        plt.ylabel(factor)
        plt.title(f"Comparison of {factor} Over Relative Time")
        plt.legend()
        plt.show()

# Example usage
file_path = 'data/BFData_02.csv'  # Replace with actual file path
analyze_brain_freeze(file_path)
