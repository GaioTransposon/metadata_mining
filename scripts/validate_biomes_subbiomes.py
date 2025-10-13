#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate biomes & sub-biomes from GPT outputs, driven by a TSV map.

Usage example (inside Docker as in your other scripts):

docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/validate_biomes_subbiomes.py \
    --map_tsv gpt_file_label_map.tsv \
    --gold_dict gold_dict.pkl \
    --embedding_models text-embedding-3-small,Qwen-Qwen3-Embedding-8B,Qwen-Qwen3-Embedding-4B,Qwen-Qwen3-Embedding-0.6B
    
Notes:
- Filenames listed in --map_tsv are resolved relative to --work_dir (default: /MicrobeAtlasProject).
- gold_dict_sbembeddings.json is expected in <work_dir>/embeddings unless overridden with --gold_embeddings_json.
- You can override paths via env vars too (WORK_DIR, SCRIPTS_DIR).
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
import glob  # NEW


# -----------------------------
# Path setup (Docker-friendly)
# -----------------------------
DEFAULT_WORK_DIR = os.environ.get("WORK_DIR", "/MicrobeAtlasProject")
DEFAULT_SCRIPTS_DIR = os.environ.get("SCRIPTS_DIR", "/app/scripts")

# Ensure project scripts are importable regardless of host paths
if DEFAULT_SCRIPTS_DIR not in sys.path:
    sys.path.append(DEFAULT_SCRIPTS_DIR)

# project imports (now relative to /app/scripts inside the container)
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
    save_figures_to_pdf, output_to_csv  # keeping import in case you reuse later
)

# -----------------------------
# Helpers
# -----------------------------
def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_") or "group"

def discover_models(embeddings_dir: str) -> List[str]:
    """
    Discover models from files like '*_sbembeddings__{model}.json' in embeddings_dir.
    """
    models = set()
    for p in glob.glob(os.path.join(embeddings_dir, '*_sbembeddings__*.json')):
        base = os.path.basename(p)
        m = re.search(r'_sbembeddings__([^/]+)\.json$', base)
        if m:
            models.add(m.group(1))
    return sorted(models)


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

        if set(expected).issubset(df.columns):
            df = df[expected]
        else:
            raise ValueError(f"TSV must provide columns {expected} (found: {list(df.columns)})")

        for c in expected:
            df[c] = df[c].astype(str).str.strip()

        df = df[~(df[expected].apply(lambda r: all((x == "" or pd.isna(x)) for x in r), axis=1))]

        if df.isna().any().any() or (df[expected] == "").any().any():
            bad = df[df[expected].isna().any(axis=1) | (df[expected] == "").any(axis=1)]
            raise ValueError(f"TSV has rows with missing values:\n{bad}")

        return df

    # Try header
    df_try = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    header_lower = [str(c).strip().lower() for c in df_try.columns]

    if header_lower == expected:
        return _clean_df(df_try)
    else:
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
# Core per-group pipeline
# -----------------------------

def run_validation_for_group(
    my_files: List[str],
    my_labels: List[str],
    group_name: str,
    gold_dict: Dict,
    embeddings_gd: Dict,
    work_dir: str,
    embedding_model: str  # NEW
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the pipeline for one group (test_type).
    Returns (results_df, stats_df) without writing CSVs.
    """
    embeddings_dir = os.path.join(work_dir, "embeddings")
    
    
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

    # Clean Label column and map to filenames
    results_biome = results_biome.copy()
    results_biome['Label'] = results_biome.index.astype(str).str.strip()

    filename_label_map = {
        label: os.path.basename(file) for file, label in file_label_map.items()
    }
    results_biome['Filename'] = results_biome['Label'].map(filename_label_map)

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
        #gpt_file_json = re.sub(r'\.txt|\.csv', '_sbembeddings.json', gpt_file)
        #gpt_json_file_path = os.path.join(embeddings_dir, gpt_file_json)
        
        gpt_file_base = re.sub(r'\.(txt|csv)$', '', gpt_file)

        
        
        gpt_file_json = f"{gpt_file_base}_sbembeddings__{embedding_model}.json"
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
            #gpt_base_file = gpt_file_json.replace('_sbembeddings.json', '')
            gpt_base_file = f"{os.path.basename(gpt_file_base)}__{embedding_model}"


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

    # move "Label" to the last column
    biomes_subbiomes = biomes_subbiomes[[c for c in biomes_subbiomes.columns if c != "Label"] + ["Label"]]

    # combine stats
    combined_stats_df = pd.concat([results_stats, results_df_stats], ignore_index=True) \
        if not results_df_stats.empty else results_stats.copy()

    biomes_subbiomes['embedding_model'] = embedding_model
    combined_stats_df['embedding_model'] = embedding_model

    return biomes_subbiomes, combined_stats_df


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process GPT outputs by groups from a TSV map.')
    parser.add_argument('--map_tsv', type=str, required=True,
                        help='TSV with columns: filename\\tlabel\\ttest_type (paths resolved relative to --work_dir).')
    parser.add_argument('--gold_dict', type=str, required=True,
                        help='Path to gold_dict.pkl (resolved relative to --work_dir unless absolute).')
    parser.add_argument('--work_dir', type=str, default=DEFAULT_WORK_DIR,
                        help=f'Working dir that contains GPT outputs and embeddings (default: {DEFAULT_WORK_DIR}).')
    parser.add_argument('--scripts_dir', type=str, default=DEFAULT_SCRIPTS_DIR,
                        help=f'Path to scripts for imports (default: {DEFAULT_SCRIPTS_DIR}).')
    parser.add_argument('--gold_embeddings_json', type=str, default=None,
                        help='Optional override for gold_dict_sbembeddings.json path. '
                             'Default: <work_dir>/embeddings/gold_dict_sbembeddings.json')
    parser.add_argument('--embedding_models', type=str, default='auto', # NEW
                        help=("Comma-separated model names to include (matching the "
                              "'*_sbembeddings__{model}.json' suffix). Use 'auto' to discover."))

    args = parser.parse_args()

    # Normalize paths relative to work_dir if given relative
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(args.work_dir, p)

    work_dir = args.work_dir
    embeddings_dir = os.path.join(work_dir, "embeddings")

    map_tsv_path = _resolve(args.map_tsv)
    gold_dict_path = _resolve(args.gold_dict)

    gold_embeddings_json = args.gold_embeddings_json or os.path.join(
        embeddings_dir, 'gold_dict_sbembeddings.json'
    )
    if not os.path.isabs(gold_embeddings_json):
        gold_embeddings_json = _resolve(gold_embeddings_json)


    # Load ground truth once
    with open(gold_dict_path, 'rb') as file:
        gold_dict = pickle.load(file)
    
    
    
    # Read TSV and form groups
    df_map = read_map_tsv(map_tsv_path)
    groups = groups_from_test_type(df_map)
    
    def _safe_model(model: str) -> str:
        return re.sub(r'[^A-Za-z0-9._-]+', '-', model)
    
    # Parse / discover models
    if args.embedding_models.strip().lower() == 'auto':
        models = discover_models(embeddings_dir)
    else:
        models = [m.strip() for m in args.embedding_models.split(',') if m.strip()]
    
    if not models:
        raise SystemExit("No embedding models found (check --embedding_models or embeddings directory).")

    for model in models:
        print(f"\n######## Running for embedding model: {model} ########\n")
    
        # choose gold embeddings for this model
        if args.gold_embeddings_json:
            # if user explicitly passes a file, use it as-is (must exist)
            gold_embeddings_json_model = _resolve(args.gold_embeddings_json)
        else:
            cand_model   = os.path.join(embeddings_dir, f'gold_dict_sbembeddings__{model}.json')
            cand_default = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')
            gold_embeddings_json_model = cand_model if os.path.exists(cand_model) else cand_default
    
        if not os.path.exists(gold_embeddings_json_model):
            raise SystemExit(
                f"Gold embeddings not found for model '{model}'. "
                f"Tried:\n  {os.path.relpath(cand_model, work_dir) if 'cand_model' in locals() else ''}\n"
                f"       {os.path.relpath(cand_default, work_dir) if 'cand_default' in locals() else ''}\n"
                f"Or pass --gold_embeddings_json explicitly."
            )
    
        embeddings_gd = load_embeddings(gold_embeddings_json_model)
    
        all_results, all_stats = [], []
    
        for group_name, files, labels in groups:
            res_df, stats_df = run_validation_for_group(
                files, labels, group_name, gold_dict, embeddings_gd, work_dir, model
            )
            all_results.append(add_separator(res_df))
            all_stats.append(add_separator(stats_df))
    
        combined_results_df = pd.concat(all_results, ignore_index=True)
        combined_stats_df = pd.concat(all_stats, ignore_index=True)
    
        # Per-model outputs (suffix with __{model})
        model_tag = _safe_model(model)
        out_results = os.path.join(work_dir, f'biome_subbiome_results__{model_tag}.csv')
        out_stats   = os.path.join(work_dir, f'biome_subbiome_stats__{model_tag}.csv')
    
        combined_results_df.to_csv(out_results, index=False)
        combined_stats_df.to_csv(out_stats, index=False)
    
        print(f"\nWrote files for '{model}':\n  {out_results}\n  {out_stats}\n")
    
    





