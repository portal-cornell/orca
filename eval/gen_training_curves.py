"""
Typical usage: 
1. Make sure `eval_path_csv` has the updated csvs
2. Edit `task_to_plot` to set tasks that you want to get training curves for
3. In the main directory for TemporalOT, run
- For matched experiments
    python -m eval.gen_training_curves -d metaworld -e matched
- For mismatched experiments
    python -m eval.gen_training_curves -d metaworld -e mismatched
"""


import pandas as pd
import os
import numpy as np
import argparse
from utils.math_utils import mean_and_se, smooth_with_pd_rolling
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', type=str, required=True, choices=['metaworld'], help='Domain name')
    parser.add_argument('-e', '--exp', type=str, required=True, choices=['mismatched', 'matched'], help='Experiment name')
    args = parser.parse_args()

    tasks_to_plot = ["Door-open", "Door-close", "Stick-push"]
    # All the tasks
    # tasks_to_plot = ["Button-press", "Door-close", "Door-open", "Window-open", "Lever-pull", "Hand-insert", "Push", "Basketball", "Stick-push", "Door-lock"]

    # Load the CSV file
    csv_file = os.path.join("eval/eval_path_csv", f"{args.domain}_{args.exp}.csv")
    df = pd.read_csv(csv_file)

    # Columns for approaches
    approaches = ["Threshold", "DTW", "OT", "TemporalOT", "ORCA"]
    if args.exp == "matched":
        approaches.append("ORCA+TOT pretrained (500k-500k)")

    for task in tasks_to_plot:
        """
        {
            approach1: [run_1_values, run_2_values, ...],
        }
        """
        task_results = {approach: [[] for _ in range(3)] for approach in approaches}

        # Iterate through each task and approach
        for index, row in df.iterrows():
            if task == row["Tasks"]:
                for approach in approaches:
                    approach_dir = row[approach]
                    run_num = int(row["Runs"])-1  # The csv starts with 1

                    for t in range(10000, 1000001, 10000):
                        
                        eval_path = os.path.join(approach_dir, "eval", f"{t}_return.npy")

                        if isinstance(eval_path, str) and os.path.exists(eval_path):
                            try:
                                with open(eval_path, 'rb') as file:
                                    return_values = np.load(file)
                                    task_results[approach][run_num].append(return_values)
                            except Exception as e:
                                print(f"Error reading {eval_path}, run_num={run_num}: {e}")

        # Calculate the mean and the confidence interval
        means = {approach: [] for approach in approaches}
        ses = {approach: [] for approach in approaches}

        for approach in approaches:
            raw_result = np.array(task_results[approach])  # shape (3, 100, 10), 3 training runs, 100 eval timesteps, 10 seeds for each eval timestep

            # Calculate the mean and confidence interval for each timestep
            for i in range(raw_result.shape[1]):
                values = raw_result[:, i, :].flatten()

                mean, se = mean_and_se(values)
                
                means[approach].append(mean)
                ses[approach].append(se)

        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', alpha=0.3)

        from .eval_constants import APPROACH_COLOR_DICT, APPROACH_NAME_TO_PLOT
        from demo.constants import MAX_PATH_LENGTH

        for approach in approaches:
            # Plot main line with confidence band
            timesteps = np.array(range(10000, 1000001, 10000))
            
            color = APPROACH_COLOR_DICT[approach]

            approach_name = APPROACH_NAME_TO_PLOT[approach]

            window_size = 5
            smoothed_means = smooth_with_pd_rolling(means[approach], window_size)
            smoothed_lower_bound = smooth_with_pd_rolling(np.array(means[approach])-np.array(ses[approach]), window_size)
            smoothed_upper_bound = smooth_with_pd_rolling(np.array(means[approach])+np.array(ses[approach]), window_size)

            # Plot mean[approach]-ses[approach]
            plt.fill_between(timesteps, smoothed_lower_bound, smoothed_upper_bound, color=color, alpha=0.2)
        
            # Plot the main line
            plt.plot(timesteps, smoothed_means, color=color, linewidth=1.5, 
                    label=approach_name)
        
        # Customize plot
        ax = plt.gca()
        ax.set_xlim([0, 1000000]) # Constrain to the shortest sequence (in case some are 2M long)
        ax.set_ylim([0, 35])
        plt.xlabel('Environment Steps', fontsize=20)
        plt.ylabel('Cumulative Return', fontsize=20)
        
        # # Put the legend out of the figure (make the legend line thicker)
        # leg = plt.legend(loc='upper left', bbox_to_anchor=(4, 4), fontsize=20, ncol=5)

        # # change the line width for the legend
        # for line in leg.get_lines():
        #     line.set_linewidth(8.0)

        plt.tight_layout()
        plt.title(task.replace("-", " ").title() + f" (ep-len={MAX_PATH_LENGTH[task.lower() + '-v2']})", fontsize=20)
        
        # Save plot
        plt_save_path = os.path.join(f"eval/eval_agg_results/eval_training_curves_{args.exp}", f"{args.domain}_{args.exp}_{task.lower()}_training_curves.png")

        plt.savefig(plt_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to {plt_save_path}")  

        