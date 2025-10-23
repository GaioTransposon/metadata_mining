#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:13:57 2025

@author: dgaio
"""



import numpy as np
import re
from scipy import stats



def time_to_seconds(t):
    """Convert strings like 18', 3' 17'' to total seconds."""
    match = re.match(r"(\d+)'(?:\s*(\d+)'')?", t.strip())
    if not match:
        raise ValueError(f"Invalid time format: {t}")
    minutes = int(match.group(1))
    seconds = int(match.group(2)) if match.group(2) else 0
    return minutes * 60 + seconds

# Input data
inline_times = ["18'", "17'", "20'", "3' 17''", "20'", "18'", "18'", "4' 43''", "17'", "20'"]
json_times = ["3' 49''", "3' 13''", "17'", "3' 13''", "11'", "18'", "18'", "17'", "17'", "17'"]




# Convert to seconds
inline_sec = np.array([time_to_seconds(t) for t in inline_times])
json_sec = np.array([time_to_seconds(t) for t in json_times])

# Compute stats
def describe(arr):
    mean = np.mean(arr)
    median = np.median(arr)
    sd = np.std(arr, ddof=1)   # sample SD
    sem = sd / np.sqrt(len(arr))
    return mean, median, sd, sem

inline_stats = describe(inline_sec)
json_stats = describe(json_sec)

# Print results
print("Inline stats:")
print(f"Mean: {inline_stats[0]:.1f} s ({inline_stats[0]/60:.2f} min)")
print(f"Median: {inline_stats[1]:.1f} s ({inline_stats[1]/60:.2f} min)")
print(f"SD: {inline_stats[2]:.1f} s ({inline_stats[2]/60:.2f} min)")
print(f"SEM: {inline_stats[3]:.1f} s ({inline_stats[3]/60:.2f} min)\n")

print("JSON stats:")
print(f"Mean: {json_stats[0]:.1f} s ({json_stats[0]/60:.2f} min)")
print(f"Median: {json_stats[1]:.1f} s ({json_stats[1]/60:.2f} min)")
print(f"SD: {json_stats[2]:.1f} s ({json_stats[2]/60:.2f} min)")
print(f"SEM: {json_stats[3]:.1f} s ({json_stats[3]/60:.2f} min)")


# Welch’s t-test (unequal variances)
t_stat, p_val = stats.ttest_ind(inline_sec, json_sec, equal_var=False)

print("T-test results (Welch’s):")
print(f"t = {t_stat:.3f}, p = {p_val:.4f}")

if p_val < 0.05:
    print("→ The difference between inline and json means is statistically significant (p < 0.05).")
else:
    print("→ No significant difference between inline and json means (p ≥ 0.05).")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    