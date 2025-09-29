#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified validator to reliably reproduce biome_subbiome_results.csv
- TSV input with columns: <filename> <label> [ignored third col]
- Computes:
    * Biome exact + lenient agreement (mean, sd) and sample size
    * Sub-biome cosine similarity summary if embeddings exist; leaves NaN otherwise
- Outputs: biome_subbiome_results.csv in --work_dir
"""

import os
import sys
import argparse
import pickle
import json
import re
import numpy as np
import pandas as pd

# Make local imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Reuse your existing helpers if available
try:
    from features_process import load_and_process_file, filter_common_keys
except ImportError:
    load_and_process_file = None
    filter_common_keys = None

# ------------------ Utilities ------------------

def safe_lenient_match(a, b):
    if pd.isna(a) or pd.isna(b):
        return False
    sa = str(a).strip().lower()
    sb = str(b).strip().lower()
    return (sa in sb) or (sb in sa)

def load_gold(work_dir):
    gold_pkl = os.path.join(work_dir, "gold_dict.pkl")
    if not os.path.exists(gold_pkl):
        raise FileNotFoundError(f"Missing gold_dict.pkl at {gold_pkl}")
    with open(gold_pkl, "rb") as f:
        gold_dict = pickle.load(f)
    gold_df = pd.DataFrame({
        "sample": list(gold_dict.keys()),
        "biome": [v[1] for v in gold_dict.values()]
    })
    return gold_dict, gold_df

def load_embeddings_json(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_biome_results(work_dir, file_label_map, gold_df):
    """
    Uses your existing load_and_process_file to create per-sample rows with columns:
      [sample, biome, gpt_biome, label, filename]
    Then computes exact/lenient means/sds and sample_size by label.
    """
    if load_and_process_file is None:
        raise ImportError("features_process.load_and_process_file not available. This simplified script expects it.")

    rows = []
    for file, label in file_label_map.items():
        full_path = os.path.join(work_dir, file)
        df = load_and_process_file(full_path, gold_df, label)  # your helper
        df = df.copy()
        df["Filename"] = os.path.basename(file)
        rows.append(df)

    big = pd.concat(rows, ignore_index=True)

    # Exact + lenient flags
    big["exact_agreement"] = (big["gpt_biome"] == big["biome"])
    big["lenient_agreement"] = big.apply(lambda r: safe_lenient_match(r["biome"], r["gpt_biome"]), axis=1)

    # Aggregate by label
    grp = big.groupby("label", dropna=False)
    out = pd.DataFrame({
        "Agreement biome (exact match)": grp["exact_agreement"].mean(),
        "biome_exact_match_mean": grp["exact_agreement"].mean(),
        "biome_exact_match_sd": grp["exact_agreement"].std(ddof=1),
        "Agreement biome (lenient match)": grp["lenient_agreement"].mean(),
        "biome_lenient_match_mean": grp["lenient_agreement"].mean(),
        "biome_lenient_match_sd": grp["lenient_agreement"].std(ddof=1),
        "sample_size": grp.size(),
    })

    # Add Filename column via reverse map (label->filename base) — if labels are unique
    label_to_basefile = {lbl: os.path.basename(f) for f, lbl in file_label_map.items()}
    out["Filename"] = [label_to_basefile.get(lbl) for lbl in out.index]
    out.index.name = "Label"

    return out.reset_index()

def safe_cosines(gd_emb, gpt_emb):
    gd = np.array(gd_emb, dtype=float)
    gt = np.array(gpt_emb, dtype=float)
    # cosine similarity
    denom = (np.linalg.norm(gd) * np.linalg.norm(gt))
    if denom == 0:
        return np.nan
    return float(np.dot(gd, gt) / denom)

def compute_subbiome_results(work_dir, file_label_map):
    """
    Best-effort sub-biome summary per file:
      Average Similarity, Median Similarity, Standard Deviation, subbiome_sample_size, 95th Percentile
    - Looks for:
        * embeddings/gold_dict_sbembeddings.json
        * embeddings/<gpt_file>_sbembeddings.json (with .txt/.csv replaced)
    - If anything is missing or yields 0 comparable samples, returns NaNs but keeps the row.
    """
    embeddings_dir = os.path.join(work_dir, "embeddings")
    gold_json = os.path.join(embeddings_dir, "gold_dict_sbembeddings.json")

    sub_rows = []
    gold = None
    if os.path.exists(gold_json):
        gold = load_embeddings_json(gold_json)

    for gpt_file, label in file_label_map.items():
        base = os.path.basename(gpt_file)
        gpt_json_name = re.sub(r'\.txt|\.csv', '_sbembeddings.json', base)
        gpt_json_path = os.path.join(embeddings_dir, gpt_json_name)

        rec = {
            "Filename": base,
            "Average Similarity": np.nan,
            "Median Similarity": np.nan,
            "Standard Deviation": np.nan,
            "subbiome_sample_size": 0,
            "95th Percentile": np.nan,
        }

        try:
            if (gold is None) or (not os.path.exists(gpt_json_path)):
                sub_rows.append(rec)
                continue

            gpt = load_embeddings_json(gpt_json_path)

            # Build common keys; optionally reuse your filter_common_keys if present
            if filter_common_keys is not None:
                gd_filt, gpt_filt = filter_common_keys(gold, gpt)
                common_keys = list(gd_filt.keys())
            else:
                common_keys = sorted(set(gold.keys()).intersection(gpt.keys()))

            if len(common_keys) == 0:
                sub_rows.append(rec)
                continue

            sims = []
            for k in common_keys:
                gd_emb = gold[k].get("embedding")
                gt_emb = gpt[k].get("embedding")
                if gd_emb is None or gt_emb is None:
                    continue
                s = safe_cosines(gd_emb, gt_emb)
                if not (np.isnan(s) or np.isinf(s)):
                    sims.append(s)

            if len(sims) == 0:
                sub_rows.append(rec)
                continue

            sims_arr = np.array(sims, dtype=float)

            rec.update({
                "Average Similarity": float(np.mean(sims_arr)),
                "Median Similarity": float(np.median(sims_arr)),
                "Standard Deviation": float(np.std(sims_arr, ddof=1)) if len(sims_arr) > 1 else 0.0,
                "subbiome_sample_size": int(len(sims_arr)),
                "95th Percentile": float(np.percentile(sims_arr, 95)),
            })

        except Exception:
            # keep NaNs on error but never drop the row
            pass

        sub_rows.append(rec)

    return pd.DataFrame(sub_rows)

def main():
    parser = argparse.ArgumentParser(description="Simplified validator for GPT biomes & sub-biomes (results only).")
    parser.add_argument("--work_dir", default=".", help="Base working directory")
    parser.add_argument("--map_file", required=True,
                        help="TSV with at least two columns: <filename> <label> (3rd col ignored)")
    parser.add_argument("--out_csv", default="biome_subbiome_results.csv",
                        help="Output CSV name (written under --work_dir)")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    map_path = os.path.join(work_dir, args.map_file)

    # Read mapping
    df_map = pd.read_csv(
        map_path,
        sep="\t",
        comment="#",
        header=None,
        names=["filename", "label", "extra"],
        usecols=[0, 1]
    )
    file_label_map = dict(zip(df_map["filename"].tolist(), df_map["label"].tolist()))

    # Gold truth
    _, gold_df = load_gold(work_dir)

    # 1) Biome agreement table (authoritative left table)
    biome_tbl = compute_biome_results(work_dir, file_label_map, gold_df)

    # 2) Sub-biome summary (best-effort; may contain NaNs)
    sub_tbl = compute_subbiome_results(work_dir, file_label_map)

    # 3) LEFT-join so we NEVER drop a row if sub-biome is missing/failed
    merged = pd.merge(biome_tbl, sub_tbl, on="Filename", how="left")

    # Add back Label->Filename mapping explicitly (Filename already present; Label is in biome_tbl)
    # And ensure columns order is friendly
    preferred_order = [
        "Label",
        "Filename",
        "Agreement biome (exact match)",
        "biome_exact_match_mean",
        "biome_exact_match_sd",
        "Agreement biome (lenient match)",
        "biome_lenient_match_mean",
        "biome_lenient_match_sd",
        "sample_size",
        "Average Similarity",
        "Median Similarity",
        "Standard Deviation",
        "subbiome_sample_size",
        "95th Percentile",
    ]
    # Keep any extra columns too
    remaining = [c for c in merged.columns if c not in preferred_order]
    merged = merged[[c for c in preferred_order if c in merged.columns] + remaining]

    out_path = os.path.join(work_dir, args.out_csv)
    merged.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
