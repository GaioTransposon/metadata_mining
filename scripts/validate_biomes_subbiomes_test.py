#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate biomes & sub-biomes from GPT outputs, driven by a TSV map.

Usage:
  python validate_biomes_subbiomes_test.py --map_tsv /Users/dgaio/MicrobeAtlasProject_Zenodo/gpt_file_label_map_mini.tsv

TSV format (tab-separated; header required; lines starting with # ignored):
filename    label   test_type
<fileA>     L1      sync - chunking no + chunksizes
<fileB>     L2      sync - chunking no + chunksizes
<fileC>     L3      sync - models
...
"""

import os
import pandas as pd
import pickle
import numpy as np
import sys
import re
import argparse
from itertools import combinations
from typing import List, Dict, Tuple

# project imports
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from features_process import (
    load_and_process_file, filter_common_keys
)
from embeddings_functions import (
    load_embeddings, compare_embeddings, create_shuffled_background_distribution, sample_by_category
)
from stats_module import (
    calculate_overlap_and_run_tests_biomes, compare_based_on_overlap_subbiomes,
    print_statistics, test_similarity_separation
)
from output_writing import (
    plot_biome_agreement, plot_actual_vs_background, plot_heatmap,
    save_figures_to_pdf, output_to_csv
)

# -----------------------------
# Paths (unchanged)
# -----------------------------
home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")
embeddings_dir = os.path.join(work_dir, "embeddings")
gold_dict_path = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")

# -----------------------------
# Helpers
# -----------------------------
def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_") or "group"

def read_map_tsv(path: str) -> pd.DataFrame:
    """
    Read TSV mapping file with columns: filename, label, test_type.
    Accepts: real header, commented header (# ...), or no header (assumes order).
    Ignores lines starting with '#'. Strips whitespace. Validates columns.
    """
    expected = ["filename", "label", "test_type"]

    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Reorder if expected columns present
        if set(expected).issubset(df.columns):
            df = df[expected]
        else:
            # If not, we can't clean this frame
            raise ValueError(f"TSV must provide columns {expected} (found: {list(df.columns)})")

        # Coerce to string & strip
        for c in expected:
            df[c] = df[c].astype(str).str.strip()

        # Drop fully empty rows
        df = df[~(df[expected].apply(lambda r: all((x == "" or pd.isna(x)) for x in r), axis=1))]

        # Validate non-empty
        if df.isna().any().any() or (df[expected] == "").any().any():
            bad = df[df[expected].isna().any(axis=1) | (df[expected] == "").any(axis=1)]
            raise ValueError(f"TSV has rows with missing values:\n{bad}")

        return df

    # --- Try 1: read assuming a real header row (commented header is skipped) ---
    df_try = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    header_lower = [str(c).strip().lower() for c in df_try.columns]

    if header_lower == expected:
        # Good, proceed
        return _clean_df(df_try)
    else:
        # The first row was likely treated as header -> re-read with header=None
        df_nohdr = pd.read_csv(
            path, sep="\t", comment="#", header=None, names=expected,
            dtype=str, keep_default_na=False
        )
        return _clean_df(df_nohdr)


def add_separator(df: pd.DataFrame) -> pd.DataFrame:
    """Append a single blank (NaN) row to visually separate groups."""
    if df is None or df.empty:
        return df
    blank = pd.DataFrame([{col: np.nan for col in df.columns}])
    return pd.concat([df, blank], ignore_index=True)



def groups_from_test_type(df: pd.DataFrame) -> List[Tuple[str, List[str], List[str]]]:
    """
    Returns [(test_type, files, labels), ...] preserving TSV order.
    """
    groups = []
    for test_type, g in df.groupby("test_type", sort=False):
        files = g["filename"].astype(str).tolist()
        labels = g["label"].astype(str).tolist()
        if not files:
            continue
        groups.append((str(test_type), files, labels))
    if not groups:
        raise ValueError("No groups could be formed from 'test_type'.")
    return groups

# -----------------------------
# Core per-group pipeline (mostly your original)
# -----------------------------
def run_validation_for_group(
    my_files: List[str],
    my_labels: List[str],
    group_name: str,
    gold_dict: Dict,
    embeddings_gd: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the pipeline for one group (test_type).
    Returns (results_df, stats_df) without writing CSVs.
    """
    file_label_map = dict(zip(my_files, my_labels))

    print("\n=== Running test_type group:", group_name, "===\n")
    for file, label in file_label_map.items():
        print(f"{os.path.basename(file)} - {label}")

    # ------- 1) Biome agreement -------
    gold_dict_df = pd.DataFrame(
        {'sample': list(gold_dict.keys()),
         'biome': [v[1] for v in gold_dict.values()]}
    )

    full_dfs = [
        load_and_process_file(os.path.join(work_dir, f), gold_dict_df, label)
        for f, label in file_label_map.items()
    ]
    full_agreement_df = pd.concat(full_dfs, ignore_index=True)
    full_agreement_df['agreement'] = (
        full_agreement_df['gpt_biome'] == full_agreement_df['biome']
    )

    lenient_agreement_df = pd.concat(full_dfs, ignore_index=True)
    lenient_agreement_df['agreement'] = lenient_agreement_df.apply(
        lambda row: (
            (str(row['biome']).strip().lower() in str(row['gpt_biome']).strip().lower()
             or str(row['gpt_biome']).strip().lower() in str(row['biome']).strip().lower())
            and not pd.isna(row['biome']) and not pd.isna(row['gpt_biome'])
        ),
        axis=1
    )

    full_result, lenient_result = plot_biome_agreement(
        full_agreement_df, lenient_agreement_df, file_label_map, work_dir
    )

    results_biome = pd.concat([
        full_result[['full match label']].rename(
            columns={'full match label': 'Agreement biome (exact match)'}
        ),
        full_result[['mean']].rename(columns={'mean': 'biome_exact_match_mean'}),
        full_result[['sd']].rename(columns={'sd': 'biome_exact_match_sd'}),

        lenient_result[['full+partial match label']].rename(
            columns={'full+partial match label': 'Agreement biome (lenient match)'}
        ),
        lenient_result[['mean']].rename(columns={'mean': 'biome_lenient_match_mean'}),
        lenient_result[['sd']].rename(columns={'sd': 'biome_lenient_match_sd'}),

        full_result[['Full Total Counts']].rename(
            columns={'Full Total Counts': 'sample_size'}
        )
    ], axis=1)

    # Build a clean Label column (from index) and map to filenames safely
    results_biome = results_biome.copy()
    results_biome['Label'] = results_biome.index.astype(str).str.strip()

    filename_label_map = {
        label: os.path.basename(file) for file, label in file_label_map.items()
    }
    results_biome['Filename'] = results_biome['Label'].map(filename_label_map)

    # Warn if any labels didn't map to a filename (prevents silent drops)
    missing_fn = results_biome[results_biome['Filename'].isna()]['Label'].tolist()
    if missing_fn:
        print("\n[WARN] These labels didn’t map to a filename (check TSV/whitespace):", missing_fn)

    # ------- 2) Sub-biome agreement -------
    results = {}
    results_sub_biome_rows = []

    gold_labels_all = {k: embeddings_gd[k]['sub-biome'] for k in embeddings_gd}
    gold_biomes_all = {k: embeddings_gd[k]['biome'] for k in embeddings_gd}

    for gpt_file in my_files:
        gpt_file_ori = gpt_file
        gpt_file_json = re.sub(r'\.txt|\.csv', '_sbembeddings.json', gpt_file)
        gpt_json_file_path = os.path.join(embeddings_dir, gpt_file_json)

        if not os.path.exists(gpt_json_file_path):
            print(f"[WARN] Missing embeddings JSON, skipping sub-biome calc for: {gpt_file_json}")
            continue

        embeddings_gpt = load_embeddings(gpt_json_file_path)

        # Filter to common keys
        filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
        print("Sample size after filtering:", len(filtered_gpt))

        if len(filtered_gpt) == 0:
            print(f"[WARN] No common keys after filtering for: {gpt_file_json}")
            continue

        # Compare embeddings
        compare_results = compare_embeddings(filtered_gd, filtered_gpt)

        # Stats
        actual_similarities = [result['cosine'] for result in compare_results.values()]
        avg_sim, median_sim, std_dev, percentiles, subbiome_sample_size = \
            print_statistics(actual_similarities)
        results[gpt_file_ori] = compare_results

        # Background comparison
        background_similarities = create_shuffled_background_distribution(
            filtered_gd, filtered_gpt, num_comparisons=len(actual_similarities)
        )
        MWU_stat, MWU_p_value = test_similarity_separation(
            actual_similarities, background_similarities
        )
        title = f"Comparison of Actual vs Background Cosine Similarity for\n{gpt_file_json}"
        comparison_fig = plot_actual_vs_background(
            actual_similarities, background_similarities, title,
            avg_sim, median_sim, std_dev, MWU_stat, MWU_p_value
        )

        # Collect row
        results_sub_biome_rows.append({
            'Average Similarity': avg_sim,
            'Median Similarity': median_sim,
            'Standard Deviation': std_dev,
            'subbiome_sample_size': subbiome_sample_size,
            '95th Percentile': percentiles,
            'MWU Statistic': MWU_stat,
            'MWU P-value': MWU_p_value,
            'Filename': gpt_file_ori,
        })

        # Heatmap (sample 10 per biome) + save per-file PDF
        common_keys = list(filtered_gd.keys() & filtered_gpt.keys())
        sampled_keys = sample_by_category(common_keys, gold_biomes_all, 10)

        if sampled_keys:
            matrix_gd = np.array([embeddings_gd[key]['embedding'] for key in sampled_keys])
            matrix_gpt = np.array([embeddings_gpt[key]['embedding'] for key in sampled_keys])
            gold_labels_sampled = {key: gold_labels_all[key] for key in sampled_keys}
            gpt_labels_sampled = {key: embeddings_gpt[key]['sub-biome'] for key in sampled_keys}

            heatmap_fig = plot_heatmap(
                matrix_gd, matrix_gpt, gpt_labels_sampled, gold_labels_sampled,
                sampled_keys, sampled_keys
            )
            gpt_base_file = gpt_file_json.replace('_sbembeddings.json', '')
            save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_base_file, embeddings_dir)
        else:
            print(f"[WARN] No sampled keys available for heatmap: {gpt_file_json}")

    results_subbiome = pd.DataFrame(results_sub_biome_rows)

    # ------- 3) Stats (biomes) -------
    results_stats = calculate_overlap_and_run_tests_biomes(full_agreement_df)
    results_stats['Filename1'] = results_stats['Label1'].map(filename_label_map)
    results_stats['Filename2'] = results_stats['Label2'].map(filename_label_map)
    results_stats['validation'] = 'biome'

    # ------- 4) Stats (sub-biomes) -------
    results_data = []
    for file1, file2 in combinations(results.keys(), 2):
        print(f"\nComparing file:\n{file1}\nwith file:\n{file2}\n")
        overlap_percentage, stat, p_value, p_adjusted, test_type = \
            compare_based_on_overlap_subbiomes(results[file1], results[file2])

        results_data.append({
            'Statistic': stat,
            'P-value': p_value,
            'Adjusted P-value': p_adjusted,
            'Test Type': test_type,
            'Filename1': file1,
            'Filename2': file2
        })
    results_df_stats = pd.DataFrame(results_data)
    if not results_df_stats.empty:
        results_df_stats['validation'] = 'sub-biome'
        reversed_filename_label_map = {v: k for k, v in filename_label_map.items()}
        results_df_stats['Label1'] = results_df_stats['Filename1'].map(reversed_filename_label_map)
        results_df_stats['Label2'] = results_df_stats['Filename2'].map(reversed_filename_label_map)

    # ------- 5) Combine (LEFT merge so biome rows are preserved) -------
    biomes_subbiomes = pd.merge(
        results_biome.reset_index(drop=True),
        results_subbiome, on='Filename', how='left'
    )

    # Prefer the TSV mapping for Label (source of truth)
    reverse_map = {v: k for k, v in filename_label_map.items()}
    biomes_subbiomes['Label'] = biomes_subbiomes['Filename'].map(reverse_map)

    # Return DataFrames (caller will concatenate across groups and write once)
    combined_stats_df = pd.concat([results_stats, results_df_stats], ignore_index=True) \
        if not results_df_stats.empty else results_stats.copy()

    # move "Label" to the last column
    biomes_subbiomes = biomes_subbiomes[[c for c in biomes_subbiomes.columns if c != "Label"] + ["Label"]]

    return biomes_subbiomes, combined_stats_df


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process GPT outputs by groups from a TSV map.')
    parser.add_argument('--map_tsv', type=str, required=True,
                        help='TSV with columns: filename\\tlabel\\ttest_type')
    args = parser.parse_args()

    # Load ground truth once
    with open(gold_dict_path, 'rb') as file:
        gold_dict = pickle.load(file)
    gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')
    embeddings_gd = load_embeddings(gold_dict_json_path)

    # Read TSV and form groups by test_type
    df_map = read_map_tsv(args.map_tsv)
    groups = groups_from_test_type(df_map)

    # Run all groups fresh each time (combine within THIS run)
    all_results, all_stats = [], []
    for group_name, files, labels in groups:
        res_df, stats_df = run_validation_for_group(files, labels, group_name, gold_dict, embeddings_gd)
        all_results.append(add_separator(res_df))
        all_stats.append(add_separator(stats_df))

    combined_results_df = pd.concat(all_results, ignore_index=True)
    combined_stats_df = pd.concat(all_stats, ignore_index=True)

    # Always overwrite outputs (bypass output_to_csv if it appends)
    out_results = os.path.join(work_dir, 'biome_subbiome_results.csv')
    out_stats   = os.path.join(work_dir, 'biome_subbiome_stats.csv')

    combined_results_df.to_csv(out_results, index=False)   # overwrite
    combined_stats_df.to_csv(out_stats, index=False)       # overwrite


    print(f"\nWrote combined files:\n  {out_results}\n  {out_stats}\n")




