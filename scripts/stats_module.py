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
from scipy.stats import fisher_exact  # <-- added for Fisher's exact

# The new variance check prevents division-by-zero whenever there is a constant group. 
# If there’s any variation at all, ttest_ind behaves fine, so no blow-ups.



def calculate_overlap_and_run_tests_biomes(df, overlap_threshold=0.9):
    sample_sets = df.groupby('label')['sample'].apply(set)
    results = []

    for label1, label2 in combinations(sample_sets.keys(), 2):
        samples1 = sample_sets[label1]
        samples2 = sample_sets[label2]
        common_samples = samples1 & samples2

        # Asymmetric overlap: how much of the smaller set is covered
        min_size = min(len(samples1), len(samples2))
        overlap_ratio = len(common_samples) / min_size

        # Extract agreement values
        df1 = df[df['label'] == label1]
        df2 = df[df['label'] == label2]

        if overlap_ratio >= overlap_threshold:
            # Perform paired test (McNemar)
            merged_df = df1.merge(df2, on='sample', suffixes=(f'_{label1}', f'_{label2}'))
            a = len(merged_df[(merged_df[f'agreement_{label1}'] == 1) & (merged_df[f'agreement_{label2}'] == 1)])
            b = len(merged_df[(merged_df[f'agreement_{label1}'] == 1) & (merged_df[f'agreement_{label2}'] == 0)])
            c = len(merged_df[(merged_df[f'agreement_{label1}'] == 0) & (merged_df[f'agreement_{label2}'] == 1)])
            d = len(merged_df[(merged_df[f'agreement_{label1}'] == 0) & (merged_df[f'agreement_{label2}'] == 0)])
            table = [[a, b], [c, d]]
            result = mcnemar(table, exact=False, correction=True)
            test_type = 'McNemar'
            stat, p = result.statistic, result.pvalue
        else:
            # Unpaired comparison:
            # If either group's binary agreement has zero variance, use Fisher's exact test.
            # Otherwise, fall back to the independent t-test as before.
            g1 = df1['agreement'].astype(float).values
            g2 = df2['agreement'].astype(float).values

            var1 = np.var(g1, ddof=1) if len(g1) > 1 else 0.0
            var2 = np.var(g2, ddof=1) if len(g2) > 1 else 0.0

            if var1 == 0.0 or var2 == 0.0:
                # Build 2x2 table: successes/failures by group
                s1 = int(np.nansum(g1))
                f1 = int(len(g1) - s1)
                s2 = int(np.nansum(g2))
                f2 = int(len(g2) - s2)
                table = [[s1, f1], [s2, f2]]

                # Fisher's exact (two-sided). Returns odds_ratio (stat) and p-value.
                odds_ratio, p = fisher_exact(table, alternative='two-sided')
                stat = odds_ratio
                test_type = 'Fisher exact'
            else:
                # Perform unpaired t-test (original behavior)
                stat, p = ttest_ind(g1, g2)
                test_type = 'Independent T-test'

        results.append((label1, label2, stat, p, test_type))

    # Apply Bonferroni correction
    corrected_results = []
    for label1, label2, stat, p_value, test_type in results:
        p_adjusted = min(p_value * len(results), 1.0)
        corrected_results.append((label1, label2, round(stat, 2), round(p_value, 5), round(p_adjusted, 5), test_type))

    results_df = pd.DataFrame(corrected_results, columns=['Label1', 'Label2', 'Statistic', 'P-value', 'Adjusted P-value', 'Test Type'])
    return results_df


def compare_based_on_overlap_subbiomes(similarities_dict1, similarities_dict2, threshold=0.8):
    keys1 = set(similarities_dict1.keys())
    keys2 = set(similarities_dict2.keys())
    common_keys = keys1 & keys2

    # Asymmetric overlap: how much of the smaller set is shared
    overlap_percentage = len(common_keys) / min(len(keys1), len(keys2))
    print(f"Percentage of overlapping samples: {overlap_percentage * 100:.2f}%")

    sorted_common_keys = sorted(common_keys)
    similarities1 = [similarities_dict1[key]['cosine'] for key in sorted_common_keys]
    similarities2 = [similarities_dict2[key]['cosine'] for key in sorted_common_keys]

    if overlap_percentage >= threshold:
        stat, p_value = ttest_rel(similarities1, similarities2)
        test_type = 'ttest_rel'
    else:
        stat, p_value = ttest_ind(similarities1, similarities2)
        test_type = 'ttest_ind'

    num_tests = len(sorted_common_keys)
    p_adjusted = min(p_value * num_tests, 1.0)

    print(f"{test_type} t-test result: t={stat:.2f}, p={p_value:.5f}, p-adj={p_adjusted:.5f}")
    
    print("#####################################################")
    print(f"Dict1 keys: {list(similarities_dict1.keys())[:5]}")
    print(f"Dict2 keys: {list(similarities_dict2.keys())[:5]}")
    print(f"Common: {len(common_keys)} / Min(len1={len(keys1)}, len2={len(keys2)}) = {overlap_percentage:.2f}")

    return overlap_percentage * 100, round(stat, 2), round(p_value, 5), round(p_adjusted, 5), test_type


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





