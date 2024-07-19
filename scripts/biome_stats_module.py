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



def calculate_overlap_and_run_tests(df):
    threshold = 0.7
    sample_sets = df.groupby('label')['sample'].apply(set)

    # Ensure there are at least two labels
    if len(sample_sets) < 2:
        return "Not enough labels to perform any comparisons."

    # Extract sample sets for the first two labels
    label1, label2 = sample_sets.index[:2]
    samples1 = sample_sets[label1]
    samples2 = sample_sets[label2]

    # Calculate the overlap and total union of samples between these two labels
    common_samples = samples1 & samples2
    total_samples = samples1 | samples2
    overlap_percentage = len(common_samples) / len(total_samples)

    results = []
    if overlap_percentage >= threshold:
        # Paired Test - McNemar's test for all label pairs
        for label1, label2 in combinations(sample_sets.keys(), 2):
            df1 = df[df['label'] == label1]
            df2 = df[df['label'] == label2]
            merged_df = df1.merge(df2, on='sample', suffixes=(f'_{label1}', f'_{label2}'))
            a = len(merged_df[(merged_df[f'agreement_{label1}'] == 1) & (merged_df[f'agreement_{label2}'] == 1)])
            b = len(merged_df[(merged_df[f'agreement_{label1}'] == 1) & (merged_df[f'agreement_{label2}'] == 0)])
            c = len(merged_df[(merged_df[f'agreement_{label1}'] == 0) & (merged_df[f'agreement_{label2}'] == 1)])
            d = len(merged_df[(merged_df[f'agreement_{label1}'] == 0) & (merged_df[f'agreement_{label2}'] == 0)])
            table = [[a, b], [c, d]]
            result = mcnemar(table, exact=False, correction=True)
            results.append((label1, label2, result.statistic, result.pvalue, 'McNemar'))
    else:
        # Independent Test - T-tests for each pair of new labels
        df['new_label'] = df['label'].str.split(',').str[0]
        new_label_values = df['new_label'].unique()
        for new_label1, new_label2 in combinations(new_label_values, 2):
            group1_data = df[df['new_label'] == new_label1]['agreement']
            group2_data = df[df['new_label'] == new_label2]['agreement']
            stat, p = ttest_ind(group1_data, group2_data)
            results.append((new_label1, new_label2, stat, p, 'Independent T-test'))

    # Apply Bonferroni correction
    correction_factor = len(results)
    corrected_results = []
    for result in results:
        label1, label2, stat, p_value, test_type = result
        corrected_pvalue = min(p_value * correction_factor, 1.0)
        corrected_results.append((label1, label2, stat, p_value, corrected_pvalue, test_type))

    results_df = pd.DataFrame(corrected_results, columns=['Label1', 'Label2', 'Statistic', 'P-value', 'Adjusted P-value', 'Test Type'])
    print(results_df)
    return results_df

