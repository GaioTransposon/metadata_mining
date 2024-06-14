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
from itertools import combinations


def mcnemar_test_with_correction(df):
    # Get unique label values
    label_values = df['label'].unique()

    # Generate all pairs of label values for comparison
    label_pairs = list(combinations(label_values, 2))
    results = []

    for label1, label2 in label_pairs:
        df1 = df[df['label'] == label1]
        df2 = df[df['label'] == label2]

        # Merge on sample to compare results
        merged_df = df1.merge(df2, on='sample', suffixes=(f'_{label1}', f'_{label2}'))

        # Create the contingency table
        a = len(merged_df[(merged_df[f'agreement_{label1}'] == 1) & (merged_df[f'agreement_{label2}'] == 1)])
        b = len(merged_df[(merged_df[f'agreement_{label1}'] == 1) & (merged_df[f'agreement_{label2}'] == 0)])
        c = len(merged_df[(merged_df[f'agreement_{label1}'] == 0) & (merged_df[f'agreement_{label2}'] == 1)])
        d = len(merged_df[(merged_df[f'agreement_{label1}'] == 0) & (merged_df[f'agreement_{label2}'] == 0)])

        # Construct the table
        table = [[a, b],
                 [c, d]]

        # Perform McNemar's test
        result = mcnemar(table, exact=True)
        results.append((label1, label2, result.statistic, result.pvalue))

    # Apply Bonferroni correction
    corrected_results = []
    correction_factor = len(results)
    for label1, label2, stat, pvalue in results:
        corrected_pvalue = min(pvalue * correction_factor, 1.0)  # Corrected p-value
        corrected_results.append((label1, label2, stat, pvalue, corrected_pvalue))
        
    print(pd.DataFrame(corrected_results, columns=['Label1', 'Label2', 'Statistic', 'P-value', 'Corrected P-value']))
    # Output the results
    return pd.DataFrame(corrected_results, columns=['Label1', 'Label2', 'Statistic', 'P-value', 'Corrected P-value'])


def t_test_agreements(df):
    # Create a new label column by splitting the string and keeping the part before the comma
    df['new_label'] = df['label'].str.split(',').str[0]

    # Get unique new labels
    new_label_values = df['new_label'].unique()

    # Perform independent t-tests for each pair of new labels
    independent_results = []
    for new_label1, new_label2 in combinations(new_label_values, 2):
        group1_data = df[df['new_label'] == new_label1]['agreement']
        group2_data = df[df['new_label'] == new_label2]['agreement']
        
        stat, p = ttest_ind(group1_data, group2_data)
        independent_results.append((new_label1, new_label2, stat, p))

    # Apply Bonferroni correction for independent tests
    correction_factor_independent = len(independent_results)
    independent_corrected_results = []
    for new_label1, new_label2, stat, p in independent_results:
        corrected_pvalue = min(p * correction_factor_independent, 1.0)
        independent_corrected_results.append((new_label1, new_label2, stat, p, corrected_pvalue))

    independent_results_df = pd.DataFrame(independent_corrected_results, columns=['NewLabel1', 'NewLabel2', 'T-statistic', 'P-value', 'Corrected P-value'])

    print("Independent t-tests results:")
    print(independent_results_df)

    return independent_results_df