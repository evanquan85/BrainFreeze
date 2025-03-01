import pandas as pd
import numpy as np
import scipy.stats as stats

def paired_t_test(normoxic_file, hypoxic_file):
    """Performs a paired t-test between normoxic and hypoxic data for each physiological factor."""
    normoxic_df = load_data(normoxic_file)
    hypoxic_df = load_data(hypoxic_file)

    # Extract relevant columns for analysis
    factors = ['MCAv_mean', 'MCAv_dia', 'MCAv_raw', 'MCAv_sys', 'FP_raw',
               'HCU_pressure', 'Systolic', 'Mean_arterial', 'Diastolic', 'HR', 'MCA_PI']

    results = []
    for factor in factors:
        clean_normoxic = normoxic_df[factor].dropna().values
        clean_hypoxic = hypoxic_df[factor].dropna().values

        # Trim longer array to match the shorter one
        min_len = min(len(clean_normoxic), len(clean_hypoxic))
        if min_len > 1:
            t_stat, p_value = stats.ttest_rel(clean_normoxic[:min_len], clean_hypoxic[:min_len])
            t_stat = round(t_stat, 4)
            p_value = round(p_value, 4)
        else:
            t_stat, p_value = np.nan, np.nan  # Assign NaN if test cannot be performed

        results.append({
            "Factor": factor,
            "T-Statistic": t_stat,
            "P-Value": p_value
        })

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  # Set output width to avoid truncation
    print(results_df)

    return results_df

# File paths for normoxic and hypoxic datasets
normoxic_file_path = 'data/BFData_02.csv'  # Replace with actual file path
hypoxic_file_path = 'data/'  # Replace with actual file path

# Run the paired t-test
paired_t_test(normoxic_file_path, hypoxic_file_path)