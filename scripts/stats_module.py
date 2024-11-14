#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:45:04 2024

@author: dgaio
"""

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from itertools import combinations
from scipy.stats import ttest_ind
import numpy as np
from scipy.stats import ttest_rel, mannwhitneyu



def calculate_overlap_and_run_tests_biomes(df):
    threshold = 0.7
    sample_sets = df.groupby('label')['sample'].apply(set)
    results = []

    # Iterate over all pairs of labels to calculate overlap and decide on the test type
    for label1, label2 in combinations(sample_sets.keys(), 2):
        samples1 = sample_sets[label1]
        samples2 = sample_sets[label2]
        common_samples = samples1 & samples2
        total_samples = samples1 | samples2
        overlap_percentage = len(common_samples) / len(total_samples)

        if overlap_percentage >= threshold:
            # Extract agreement data for each label
            df1 = df[df['label'] == label1]
            df2 = df[df['label'] == label2]
            # Merge on 'sample' column
            merged_df = df1.merge(df2, on='sample', suffixes=(f'_{label1}', f'_{label2}'))
            # Construct the contingency table for McNemar's test
            a = len(merged_df[(merged_df[f'agreement_{label1}'] == 1) & (merged_df[f'agreement_{label2}'] == 1)])
            b = len(merged_df[(merged_df[f'agreement_{label1}'] == 1) & (merged_df[f'agreement_{label2}'] == 0)])
            c = len(merged_df[(merged_df[f'agreement_{label1}'] == 0) & (merged_df[f'agreement_{label2}'] == 1)])
            d = len(merged_df[(merged_df[f'agreement_{label1}'] == 0) & (merged_df[f'agreement_{label2}'] == 0)])
            table = [[a, b], [c, d]]
            # Perform McNemar's test
            result = mcnemar(table, exact=False, correction=True)
            results.append((label1, label2, result.statistic, result.pvalue, 'McNemar'))
        else:
            # Compare agreements using independent t-test
            group1_data = df[df['label'] == label1]['agreement']
            group2_data = df[df['label'] == label2]['agreement']
            stat, p = ttest_ind(group1_data, group2_data)
            results.append((label1, label2, stat, p, 'Independent T-test'))

    # Apply Bonferroni correction
    correction_factor = len(results)
    corrected_results = []
    for result in results:
        label1, label2, stat, p_value, test_type = result
        corrected_pvalue = min(p_value * correction_factor, 1.0)
        corrected_results.append((label1, label2, round(stat, 2), round(p_value, 5), round(corrected_pvalue, 5), test_type))

    # Prepare results DataFrame
    results_df = pd.DataFrame(corrected_results, columns=['Label1', 'Label2', 'Statistic', 'P-value', 'Adjusted P-value', 'Test Type'])
    return results_df




def compare_based_on_overlap_subbiomes(similarities_dict1, similarities_dict2, threshold=0.7):
    keys1 = set(similarities_dict1.keys())
    keys2 = set(similarities_dict2.keys())
    common_keys = keys1 & keys2
    total_keys = keys1 | keys2
    overlap_percentage = len(common_keys) / len(total_keys)
    print('Percentage of overlapping samples: ', overlap_percentage*100)
    
    sorted_common_keys = sorted(common_keys)
    similarities1 = [similarities_dict1[key]['cosine'] for key in sorted_common_keys]
    similarities2 = [similarities_dict2[key]['cosine'] for key in sorted_common_keys]

    if overlap_percentage >= threshold:
        stat, p_value = ttest_rel(similarities1, similarities2)
        test_type = 'ttest_rel'
    else:
        stat, p_value = ttest_ind(similarities1, similarities2)
        test_type = 'ttest_ind'

    num_tests=len(sorted_common_keys)
    p_adjusted = min(p_value * num_tests, 1.0)  # ensures p-value does not exceed 1
    
    print(f"{test_type} t-test result: t={stat}, p={p_value}, p-adj={p_adjusted}")
    return overlap_percentage*100, round(stat, 2), round(p_value, 5), round(p_adjusted, 5), test_type



def print_statistics(similarities):
    
    avg_sim = round(np.mean(similarities), 2)
    median_sim = round(np.median(similarities), 2)
    std_dev = round(np.std(similarities), 2)
    percentiles = np.percentile(similarities, [25, 50, 75])
    percentiles = np.round(percentiles, 2)  
    subbiome_sample_size = len(similarities)
    
    print(f"Average cosine similarity: {avg_sim}")
    print(f"Median cosine similarity: {median_sim}")
    print(f"Standard deviation of cosine similarity: {std_dev}")
    print(f"Percentiles: {percentiles[0]}, {percentiles[1]}, {percentiles[2]}")
    print(f"How many similarities: {subbiome_sample_size}")
    
    
    return avg_sim, median_sim, std_dev, percentiles, subbiome_sample_size





def test_similarity_separation(actual_similarities, background_similarities):
    """Performs a statistical test to see if actual and background similarities are significantly different and returns the p-value."""
    stat, p_value = mannwhitneyu(actual_similarities, background_similarities)
    print(f"Actual vs background similarities: Mann-Whitney U test: U={stat}, p-value={p_value}")
    return round(stat, 2), round(p_value, 3)







