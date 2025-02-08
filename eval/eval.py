
"""
Usage to plot workshop results

Create plots for the workshop paper's joint-based distance metrics
    python eval_performance.py -w

Create plots for the workshop paper's visual-based distance metrics
    python eval_performance.py -w -v
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import List, Tuple
import os
import glob
import yaml
import numpy as np
import scipy.stats as stats
from torchvision.utils import save_image
import argparse
import pandas as pd
from utils.math_utils import interquartile_mean_and_ci

def extract_timestep(filename: str) -> int:
    """Extract timestep from filename of format '{timestep}_rollouts_geom_xpos_states.npy'"""
    match = re.match(r'(\d+)_rollouts_geom_xpos_states\.npy', Path(filename).name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid filename format: {filename}")

def extract_exp_label_from_dir(exp_dir):
    return Path(exp_dir).name.split('=')[-1]


def plot_multiple_directories(directory_results, 
                            output_file: str = 'multi_directory_performance.png',
                            labels: List[str] = [],
                            title: str = "",
                            smoothing=5,
                            clip_intervals=False):
    """
    Create and save plot comparing performance across multiple directories
    
    Args:
        directory_results: Dictionary mapping directory names to (timesteps, performances) tuples
        output_file: Path to save the output plot
        clip_intervals: True to plot the intervals in (0.0, 1.0)
    """
    
    # Get a good color palette for the number of directories
    colors = plt.get_cmap('Dark2').colors
    
    min_last_timestep = float('inf')

    # Hack to make zip work well
    if labels:
        label_ids = list(range(len(labels)))
    else:
        label_ids = list(range(len(directory_results)))
    
    # Plot each directory's data
    for (dir_name, (performances, lower, upper, timesteps)), color, label_id in zip(directory_results.items(), colors, label_ids):
        # Plot main line with confidence band
        timesteps = np.array(timesteps)
        performances_smooth = smooth(np.array(performances), alpha=smoothing)
        lower_smooth = smooth(np.array(lower), alpha=smoothing)
        upper_smooth = smooth(np.array(upper), alpha=smoothing)
        
        # Make sure it always starts at 0
        performances_smooth[0] = performances[0]
        lower_smooth[0] = lower[0]
        upper_smooth[0] = upper[0]

        if clip_intervals:
            lower_smooth = np.clip(lower_smooth, a_min=0.0, a_max=1.0)
            upper_smooth = np.clip(upper_smooth, a_min=0.0, a_max=1.0)

        # Plot confidence interval
        plt.fill_between(timesteps, lower_smooth, upper_smooth, color=color, alpha=0.2)
      
        # Plot the main line
        if labels:
            exp_label = labels[label_id]
        else:
            exp_label = extract_exp_label_from_dir(dir_name)
        plt.plot(timesteps, performances_smooth, color=color, linewidth=1.5, 
                label=exp_label, alpha=0.8)
       # plt.ylim(0,1)

        min_last_timestep = min(max(timesteps), min_last_timestep)
    
    # Customize plot
    ax = plt.gca()
    ax.set_xlim([0, min_last_timestep]) # Constrain to the shortest sequence (in case some are 2M long)

    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title(title if title else 'Geometric State Performance Comparison', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust legend
    plt.legend()
    

    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as {output_file}") 


def smooth(x, alpha:int):
    if alpha > 1:
        """
        smooth data with moving window average.
        that is, smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(alpha)
        z = np.ones(len(x))
        smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
        return smoothed_x
    return x

def compute_performance(rollout_directory, performance_col="eval/final_success_rate"):
    """
    Options for performance_col are generally final_success_rate and total_success_rate
    """
    
    df = pd.read_csv(rollout_directory)
    filtered_df = df.fillna(0, inplace=True)

    # Extract the relevant columns
    performance_array = filtered_df[performance_col].to_numpy()
    step_array = filtered_df["step"].to_numpy()

    iqms = []
    cis_lower = []
    cis_upper = []
    for performance in performance_array:
        
        iqm, ci_lower, ci_upper = interquartile_mean_and_ci(performance)
        iqms.append(iqm)
        cis_lower.append(ci_lower)
        cis_upper.append(ci_upper)

    return iqms, cis_lower, cis_upper, step_array

def compute_performance_many_experiments(rollout_directories, performance_col):
    all_rollout_performances = {}
    for rollout_directory in rollout_directories:
        if not rollout_directory: # may have empty directories (if for example an experiment has not finished yet)
            continue
        print(f"Computing performance for {rollout_directory}")
        iqms, cis_lower, cis_upper, timesteps = compute_performance(rollout_directory, performance_col)
        all_rollout_performances[rollout_directory] = (iqms,cis_lower, cis_upper, timesteps)
    
    return all_rollout_performances

FULL_EXPERIMENTS = {
    "button-press-v2": {
        "Coverage": [], # put runs here
        "TemporalOT": []
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default="button-press-v2", help="Specify which task to run experiment for")    

    args = parser.parse_args()
    task = args.task
    plot_folder = "icml_figs"
    plot_smoothing = 5
    base_exp_dir = "train_logs/"
    performance_col = "eval/total_success_rate"
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)


    if task == "all":
        task_set = list(FULL_EXPERIMENTS.keys())
    else:
        task_set = [task]

    for task in task_set:
        exp_folder = FULL_EXPERIMENTS
        task_experiments = exp_folder[task]
        reward_types = list(task_experiments.keys())
        
        all_task_results = {}
        for reward_type in reward_types:
            reward_performances = []
            for run in task_experiments[reward_type]:
                
                df = pd.read_csv(os.path.join(base_exp_dir, run, "eval", "performance.csv"))
                df.fillna(0, inplace=True)

                # Extract the relevant columns
                performance_array = df[performance_col].to_numpy()
                step_array = df["step"].to_numpy()

                reward_performances.append(performance_array)
            reward_performances = np.stack(reward_performances)
            mean_performance = reward_performances.mean(axis=0)
            std_performance = reward_performances.std(axis=0)
            lower = mean_performance - std_performance
            upper = mean_performance + std_performance
            all_task_results[reward_type] = (mean_performance, lower, upper, step_array)
        
        plot_file = os.path.join(plot_folder, f"{task}.png")
        plot_multiple_directories(all_task_results, plot_file, reward_types, title=task)

