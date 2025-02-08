import numpy as np
import torch
import ot
import numpy as np
from scipy.special import logsumexp
import scipy.stats as stats

def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin))**2, 2))
    return c

def interquartile_mean_and_ci(values, confidence=0.95):
    # Sort the array
    sorted_values = np.sort(values)
    
    # Calculate the first and third quartile
    Q1 = np.percentile(sorted_values, 25)
    Q3 = np.percentile(sorted_values, 75)
    
    # Get the values between Q1 and Q3 (inclusive)
    interquartile_values = sorted_values[(sorted_values >= Q1) & (sorted_values <= Q3)]
    
    # Compute the interquartile mean
    interquartile_mean = np.mean(interquartile_values)
    
    # Compute the sample mean and standard error of the mean (SEM)
    sample_mean = np.mean(values)
    sem = stats.sem(values)  # Standard Error of the Mean
    
    # Compute the margin of error for the 95% confidence interval
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(values)-1)
    
    # Compute the confidence interval
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return interquartile_mean, ci_lower, ci_upper

def interquartile_mean_and_se(values):
    # Sort the array
    sorted_values = np.sort(values)
    
    # Calculate the first and third quartile
    Q1 = np.percentile(sorted_values, 25)
    Q3 = np.percentile(sorted_values, 75)
    
    # Get the values between Q1 and Q3 (inclusive)
    interquartile_values = sorted_values[(sorted_values >= Q1) & (sorted_values <= Q3)]
    
    # Compute the interquartile mean
    interquartile_mean = np.mean(interquartile_values)
    
    # Compute the standard error of the mean
    sem = np.std(interquartile_values) / np.sqrt(len(interquartile_values))

    return interquartile_mean, sem

def mean_and_std(values):
    mean = np.mean(values)
    std = np.std(values)
    return mean, std

def mean_and_se(values):
    mean = np.mean(values)
    se = np.std(values) / np.sqrt(len(values))
    return mean, se

def mean_and_ci(values, confidence=0.95):
    """Calculate the mean and confidence interval of a list of values"""
    sample_mean = np.mean(values)
    sem = stats.sem(values)  # Standard Error of the Mean
    
    # Compute the margin of error for the 95% confidence interval
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(values)-1)
    
    # Compute the confidence interval
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return sample_mean, ci_lower, ci_upper

def smooth_with_pd_rolling(data, window_size):
    import pandas as pd
    data = pd.Series(data)
    return data.rolling(window=window_size).mean()