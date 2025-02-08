"""
Typical usage: 
1. Make sure `eval_path_csv` has the updated csvs
2. In the main directory for TemporalOT, run
- For matched experiments
    python -m eval.gen_eval_results -d metaworld -e matched
- For mismatched experiments
    python -m eval.gen_eval_results -d metaworld -e mismatched
"""

import pandas as pd
import os
import numpy as np
import argparse
from utils.math_utils import interquartile_mean_and_se, mean_and_se
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', type=str, required=True, choices=['metaworld'], help='Domain name')
    parser.add_argument('-e', '--exp', type=str, required=True, choices=['mismatched', 'matched'], help='Experiment name')
    args = parser.parse_args()

    # Load the CSV file
    csv_file = os.path.join("eval/eval_path_csv", f"{args.domain}_{args.exp}.csv")
    df = pd.read_csv(csv_file)

    # Columns for approaches
    # approaches = ["Threshold", "RoboCLIP", "DTW", "OT", "TemporalOT", "ORCA"]
    if args.exp == "matched":
        approaches = ["Threshold", "DTW", "OT", "TemporalOT", "ORCA", "ORCA+TOT pretrained (500k-500k)"]
    else:
        approaches = ["Threshold", "RoboCLIP", "DTW", "OT", "TemporalOT", "ORCA", "ORCA+TOT pretrained (500k-500k)"]
    approaches = ["Threshold", "RoboCLIP", "DTW", "OT", "TemporalOT", "ORCA", "ORCA+TOT pretrained (500k-500k)"]
    # if args.exp == "matched":
    #     approaches.append("ORCA+TOT pretrained (500k-500k)")

    # Initialize a dictionary to store results
    results = {}

    # Iterate through each task and approach
    for index, row in df.iterrows():
        task_key = (row['Difficulty Level'], row['Tasks'])

        # Initialize storage for the task if not already present
        if task_key not in results:
            results[task_key] = {approach: [] for approach in approaches}

        for approach in approaches:
            path = row[approach]
            if isinstance(path, str) and os.path.exists(path):
                # Assume each folder contains a file named `results.txt` with a single float value
                if approach == "ORCA+TOT pretrained (500k-500k)":
                    final_eval_path = os.path.join(path, "eval", "500000_return.npy")
                else:
                    final_eval_path = os.path.join(path, "eval", "1000000_return.npy")

                try:
                    with open(final_eval_path, 'rb') as file:
                        return_values = np.load(file)
                        results[task_key][approach].append(return_values)
                except Exception as e:
                    print(f"Error reading {final_eval_path}: {e}")
            else:
                print(f"Path {path} does not exist for {approach}")

    """##################################################################################

        Generate the aggregated table

    ##################################################################################"""

    # Calculate mean and standard error for each task and approach
    aggregated_results = []
    for task_key, approaches_data in results.items():
        difficulty, task = task_key
        curr_task_results = {"Difficulty Level": difficulty, "Task": task}

        for approach, values in approaches_data.items():
            if values:  # Only calculate if there are valid values
                flatten_values = np.concatenate(values)
                mean_val = np.mean(flatten_values)
                # Standard error is the standard deviation divided by the square root of the number of samples
                se_val = np.std(flatten_values) / np.sqrt(len(flatten_values))
            else:
                mean_val = -1
                se_val = -1

            curr_task_results[approach] = f"{mean_val:.2f} ({se_val:.2f})"
        
        aggregated_results.append(curr_task_results)

    total_task_results = {"Difficulty Level": "Total", "Task": "Total"}

    for approach in approaches:
        all_values = [value for task_values in results.values() for value in task_values[approach]]

        if all_values:
            flatten_values = np.concatenate(all_values)
            mean_val = np.mean(flatten_values)
            # Standard error is the standard deviation divided by the square root of the number of samples
            se_val = np.std(flatten_values) / np.sqrt(len(flatten_values))
        else:
            mean_val = -1
            se_val = -1

        total_task_results[approach] = f"{mean_val:.2f} ({se_val:.2f})"

    aggregated_results.append(total_task_results)

    # Convert aggregated results to a DataFrame
    aggregated_df = pd.DataFrame(aggregated_results)

    # Save the aggregated results to a new CSV
    output_csv = os.path.join("eval/eval_agg_results", f"{args.domain}_{args.exp}_agg_result.csv")
    aggregated_df.to_csv(output_csv, index=False)

    print(f"Aggregated results saved to {output_csv}")

    """##################################################################################

        Plot the IQM with standard error

    ##################################################################################"""

    # Plotting (the IQM for all the approaches)
    # Set the grid to be under the bars
    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

    from .eval_constants import APPROACH_COLOR_DICT, APPROACH_NAME_TO_PLOT

    # from demo.constants import MAX_PATH_LENGTH

    # # For each task, normalize the cumulative return by the maximum path length
    # for task_key, approaches_data in results.items():
    #     for approach, values in approaches_data.items():
    #         if values:
    #             max_path_length = MAX_PATH_LENGTH[task_key[1].lower() + "-v2"]
    #             results[task_key][approach] = [value / max_path_length for value in values]

    for approach in approaches:
        all_values = [value for task_values in results.values() for value in task_values[approach]]

        if all_values:
            flatten_values = np.concatenate(all_values)
            
            # Calculate the interquartile mean (IQM)
            # iqm, sem = interquartile_mean_and_se(flatten_values)
            mean, se = mean_and_se(flatten_values)
        else:
            # iqm, sem = 0, 0
            mean, se = 0, 0

        approach_name = APPROACH_NAME_TO_PLOT[approach]

        # Plot the bar (with label's font size at 18)
        plt.bar(approach_name, mean, yerr=se, color=APPROACH_COLOR_DICT[approach], zorder=3, capsize=10)

        # Add the IQM value above the bar
        plt.text(approach_name, mean + 0.005, f"{mean:.2f}", ha='center', va='bottom', fontsize=16)

    plt.xticks(fontsize=16)
    plt.ylim([0, 17])

    plt.ylabel('Mean Cumulative Return', fontsize=20)

    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(f"eval/eval_agg_results/{args.domain}_{args.exp}_mean.png"), dpi=300, bbox_inches='tight')
    plt.close()


