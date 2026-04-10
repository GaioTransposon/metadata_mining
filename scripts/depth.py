#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 16:19:09 2025

@author: dgaio
"""

import gzip
import io
import re
from pathlib import Path
import csv
from collections import Counter
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


###############################################################################

# =============================================================================
# # # What happens in PART 0: 
#     
# # input: sample.info.gz
# 
# # steps: 
# #     - detect sample ids
# #     - catch all depth -containing fields 
# #     - counts how many times each field name appears 
# #     - sorts them by frequency 
# #      - print summary to console
# 
# # output: depth_containing_fields_report.txt
# =============================================================================


# --- CONFIG ---
gz_path = Path("/Users/dgaio/MicrobeAtlasProject/sample.info.gz")
report_path = gz_path.with_name("depth_containing_fields_report.txt")

# Match any key that contains 'depth' before '=' (case-insensitive)
pattern = re.compile(r'^\s*([^=\s]*depth[^=\s]*)\s*=', re.IGNORECASE)

# --- MAIN ---
counts = Counter()

with gzip.open(gz_path, "rb") as f:
    for raw in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
        m = pattern.match(raw)
        if not m:
            continue
        key = m.group(1)
        counts[key] += 1

total = sum(counts.values())

# Sort by descending count, then alphabetically for ties
sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))

# --- Print summary to console ---
print(f"Total depth-like fields seen: {total}")
for k, n in sorted_items:
    print(f"{n}\t{k}")

# --- Write full report to file ---
with report_path.open("w", encoding="utf-8") as out:
    out.write(f"Total depth-like fields seen: {total}\n")
    for k, n in sorted_items:
        out.write(f"{n}\t{k}\n")

print(f"\n✅ Wrote report to: {report_path}")


###############################################################################

# =============================================================================
# # PART 0.5: we manually inspect the file and decide:
# #     - which fields we keep
# #     - group the fields into three groups:
# #             generic
# #             marine
# #             local_surface 
# 
# 
# # Decided class with the help from GPT. We decide which class. 
# # nb: not going to make any decision on conversion using this. This is just for stats and to report the “class” in the interactive map we produce. 
# # We don’t include any field that has less then 50 occurrences. 
# =============================================================================

    
###############################################################################



# =============================================================================
# # # What happens in PART 1: 
#     
# # input: sample.info.gz
# 
# # steps: 
# #     - detect sample ids
# #     - catch field names (must be in lists)
# #     - map fields to 3 lists 
# #     - store values and tag them into classes: `missing`, `sea_level`, `mbsl`, `mbsf`, `bgs`, metric/imperial (`mm`, `cm`, `m`, `km`, `feet`, `inches`), `bare_number`, `range_or_composite`, or `other`
# #     - count occurrences of raw + per group (class), and per unit type 
# #     - write all records to `depth_data.csv.gz`
# #     - print stats to console
# 
# 
# # output: depth_data.csv.gz
# =============================================================================


# PART 1: 


# -------------------------------------------------------------------
# LISTS:

generic = [
"sample_depth",
"sample_Depth",
"sample_sd_depth_m",
"experiment_depth",
"sample_depth_m",
"sample_depth_(m)",
"sample_depth_meters",
"sample_sample_depth",
"sample_depth_sample",
"sample_Depth_m",
"sample_water_depth",
"sample_Depth_cm",
"sample_depth_(cm)",
"sample_collection_depth",
"sample_*depth",
"sample_mean_depth",
"sample_depth_cm",
"sample_sample_depth_(m)",
"sample_sample_depth/m",
"sample_depth_(meters)",
"sample_Depth(m)",
"sample_Sample_Depth",
"sample_DEPTH",
"sample_site_depth"]

marine = [
"sample_tot_depth_water_col",
"sample_secchi_depth",
"sample_bottom_depth",
"sample_total_depth_water_col",
"sample_reef_depth",
"sample_depth_below_sealevel",
"sample_StationDepth",
"sample_station_depth"]

local_surface = [
"sample_core_depth",
"sample_sediment_depth",
"sample_Perm.depth",
"sample_soil_depth_cm",
"sample_soil_depth",
"sample_Depth_cm_below_surface",
"sample_depth_belowground",
"sample_soil_depth_profile_cm",
"sample_soil_depth_inches"]

# -------------------------------------------------------------------

# --- CONFIG ---
gz_path = Path("/Users/dgaio/MicrobeAtlasProject/sample.info.gz")
out_path = gz_path.with_name("depth_data.csv.gz")

# capture key=value lines
kv_re = re.compile(r'^\s*([^=\s]+)\s*=\s*(.*)$')
# block header lines like ">SRS123456", ">ERS676566", etc.
header_re = re.compile(r'^\s*>\s*([A-Za-z0-9_.:-]+)\s*$')

# --- prepare exact-name lookup and group mapping ---
try:
    # Build key -> group map (strict equality on the raw key as seen in the file)
    group_map = {}
    for k in generic:
        group_map[k] = "generic"
    for k in marine:
        group_map[k] = "marine"
    for k in local_surface:
        group_map[k] = "local_surface"
except NameError as e:
    raise RuntimeError(
        "The lists 'generic', 'marine', and 'local_surface' must be defined above this script."
    ) from e

allowed_keys = set(group_map.keys())

def classify_value_units(v: str) -> str:
    s = v.strip().lower()

    # common 'not available'
    if s in {
        '', 'na', 'n/a', 'null', 'none', 'unknown',
        'missing', 'not collected', 'not applicable'
    }:
        return 'missing'

    # Sometimes depth fields might contain "sea level" text
    if re.fullmatch(r'\s*sea\s+level\s*\.?,?\s*', s) or re.fullmatch(r'\s*nivel\s+del\s+mar\s*\.?,?\s*', s):
        return 'sea_level'

    # marine/terrestrial depth shorthand
    if 'mbsl' in s or 'm bsl' in s:   # meters below sea level
        return 'mbsl'
    if 'mbsf' in s or 'm bsf' in s:   # meters below seafloor
        return 'mbsf'
    if 'bgs' in s:                    # below ground surface
        return 'bgs'


    # --- imperial split (feet vs inches), allow glued numbers: 5ft, 12inches
    if re.search(r'\b(ft|feet)\b', s) or re.search(r'\d(?:[.,]\d+)?\s*ft\b', s) or re.search(r'(?<=\d)ft\b', s):
        return 'feet'
    if re.search(r'\b(inch|inches)\b', s) or re.search(r'\d(?:[.,]\d+)?\s*inches\b', s) or re.search(r'(?<=\d)inches\b', s):
        return 'inches'

    # --- metric units with or without space (glued forms like 15cm, 2m, 0.1km)
    if (re.search(r'\b(km|kilometer|kilometre|kilometers|kilometres)\b', s)
        or re.search(r'(?<=\d)km\b', s)
        or re.search(r'\d(?:[.,]\d+)?\s*km\b', s)):
        return 'km'

    if (re.search(r'\b(cm|centimeter|centimetre|centimeters|centimetres)\b', s)
        or re.search(r'(?<=\d)cm\b', s)
        or re.search(r'\d(?:[.,]\d+)?\s*cm\b', s)):
        return 'cm'


    if (re.search(r'\b(mm|millimeter|millimetre|millimeters|millimetres)\b', s)
        or re.search(r'(?<=\d)mm\b', s)
        or re.search(r'\d(?:[.,]\d+)?\s*mm\b', s)):
        return 'mm'
    
    
    # meters/metres keywords OR number + m (glued or spaced)
    if (re.search(r'\b(m|meter|metre|meters|metres)\b', s)
        or re.search(r'(?<=\d)m\b', s)
        or re.search(r'\d(?:[.,]\d+)?\s*m\b', s)):
        return 'meters'

    # plain number (allow , or . as decimal/thousands)
    if re.fullmatch(r'\s*-?\d+(?:[.,]\d+)?\s*', s):
        return 'bare_number'

    # ranges like "100-200 m" or "100 to 200m"
    if re.search(r'-| to ', s):
        return 'range_or_composite'

    return 'other'



# --- counters & buffers ---
raw_key_counts = Counter()
group_counts = Counter()
unit_counts = Counter()

records = []
current_sample_id = None
unknown_sid_skipped = 0

# --- parse ---
with gzip.open(gz_path, "rb") as f:
    for raw in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
        # 1) detect block header lines like ">SRS123456"
        hm = header_re.match(raw)
        if hm:
            current_sample_id = hm.group(1)
            continue

        # 2) normal key=value parsing
        m = kv_re.match(raw)
        if not m:
            continue

        key, val = m.group(1), m.group(2)

        # STRICT exact-name filter (no regex, no normalization)
        if key not in allowed_keys:
            continue

        grp = group_map[key]
        raw_key_counts[key] += 1
        group_counts[grp] += 1

        uclass = classify_value_units(val)
        unit_counts[uclass] += 1

        if current_sample_id is None:
            # no header seen yet for this block; skip recording to avoid empty sample_id rows
            unknown_sid_skipped += 1
            continue

        # keep a "normalized_key" for downstream compatibility (lowercased exact key)
        records.append({
            "sample_id": current_sample_id,
            "group": grp,
            "normalized_key": key.lower(),
            "raw_key": key,
            "value": val.strip(),
            "unit_class": uclass,
        })

# --- SAVE OUTPUT ---
with gzip.open(out_path, "wt", encoding="utf-8", newline="") as gz_out:
    writer = csv.DictWriter(
        gz_out,
        fieldnames=["sample_id", "group", "normalized_key", "raw_key", "value", "unit_class"]
    )
    writer.writeheader()
    writer.writerows(records)

# --- REPORT ---
print("Top raw keys (exact matches):")
for k, n in raw_key_counts.most_common(100):
    print(f"{n}\t{k}")

print("\nCounts by group:")
for g, n in group_counts.most_common():
    print(f"{n}\t{g}")

print("\nValue/unit style histogram (rough):")
for u, n in unit_counts.most_common():
    print(f"{n}\t{u}")

print(f"\n✅ Saved {len(records):,} depth-related entries to {out_path}")
if unknown_sid_skipped:
    print(f"ℹ️ Skipped {unknown_sid_skipped} lines with depth keys before any '>SRS...' header was seen.")



###############################################################################


# =============================================================================
# # # What happens in PART 2: 
#     
# 
# # input: depth_data.csv.gz (from Part 1)
# 
# # steps: 
# #     - extraction of numbers from strings
# #     - parsing numbers (commas, dots)
# #     - keeps meters, bare_number, mbsf, mbsl, bgs as metres 
# #     - converts feet, inches, km, cm, mm to meters 
# #     - discard ranges or composites (like “5–10 m”)
# #     - plausibility filter: depth_m < 0 or > 12,000 m
# #     - equals sea_level to 0 m 
# #     - prints stats to console and writes reports 
# #     - saves to files
#     
# # outputs: 
# #     - depth_clean.csv.gz — standardized numeric depths
# #     - depth_report.json — summary statistics
# #     - depth_missing_values.tsv, depth_other_values.tsv — inventories of unresolved entries
# =============================================================================
    
    
    


# PART 2


# --- paths ---
in_path = Path("/Users/dgaio/MicrobeAtlasProject/depth_data.csv.gz")
out_clean_path = in_path.with_name("depth_clean.csv.gz")
out_report_path = in_path.with_name("depth_report.json")
out_missing_values_tsv = in_path.with_name("depth_missing_values.tsv")
out_other_values_tsv = in_path.with_name("depth_other_values.tsv")

# --- helpers ---

_num_re = re.compile(r"[+-]?\d[\d.,]*")

def _parse_num(s: str):
    """Extract first number from a string; handle decimal comma."""
    if not isinstance(s, str):
        return None
    m = _num_re.search(s)
    if not m:
        return None
    x = m.group(0)
    # If there is a comma but no dot, treat comma as decimal separator
    if ',' in x and '.' not in x:
        x = x.replace(',', '.')
    # Remove any remaining thousands separators
    x = x.replace(',', '')
    try:
        return float(x)
    except Exception:
        return None


def to_meters_depth(value_str: str, unit_class: str):
    """
    Convert depth to meters. Returns float or None if not convertible.

    Convertible classes:
      - sea_level -> 0.0
      - meters, bare_number, cm, km
      - feet, inches
      - mbsf (meters below seafloor), mbsl (meters below sea level), bgs (below ground surface)
    Non-convertible:
      - missing, other, range_or_composite
    """
    unit_class = (unit_class or "").lower()

    if unit_class == "sea_level":
        return 0.0

    if unit_class in ("missing", "other", "range_or_composite"):
        return None

    num = _parse_num(value_str)
    if num is None:
        return None

    if unit_class in ("meters", "bare_number", "mbsf", "mbsl", "bgs"):
        # These are already in meters (semantically "depth below …")
        return num
    if unit_class == "feet":
        return num * 0.3048
    if unit_class == "inches":
        return num * 0.0254
    if unit_class == "km":
        return num * 1000.0
    if unit_class == "cm":
        return num / 100.0
    if unit_class == "mm":                      
        return num / 1000.0                   

    # Fallback: assume meters if numeric
    return num




# --- load ---
df = pd.read_csv(in_path)

# Sanity: ensure required columns exist
required_cols = {"sample_id", "normalized_key", "raw_key", "value", "unit_class"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise SystemExit(f"Input file missing required columns: {missing_cols}")

# --- drop ranges ---
is_range = df["unit_class"].str.lower().eq("range_or_composite")
df_norange = df.loc[~is_range].copy()

# --- convert to meters (depth) ---
df_norange["depth_m"] = df_norange.apply(
    lambda r: to_meters_depth(r["value"], r["unit_class"]), axis=1
)

# Keep rows with numeric depth
is_num = df_norange["depth_m"].notna()

# We'll also capture what's missing/other/unparsed for reporting
is_missing = df_norange["unit_class"].str.lower().eq("missing")
is_other = df_norange["unit_class"].str.lower().eq("other")
is_unparsed_but_expected = (~is_range) & (~is_num) & (~is_missing) & (~is_other)

# --- save cleaned dataset ---
# Keep group if present (produced in PART 1), otherwise omit
base_cols = ["sample_id", "depth_m", "normalized_key", "raw_key", "value", "unit_class"]
if "group" in df_norange.columns:
    clean_cols = ["sample_id", "group", "depth_m", "normalized_key", "raw_key", "value", "unit_class"]
else:
    clean_cols = base_cols

df_clean = df_norange.loc[is_num, clean_cols].copy()

# --- plausibility filter for depth ---
# Depth should be >= 0 and not absurdly large (cap at 12,000 m).
plausible = df_clean["depth_m"].between(0.0, 12000.0, inclusive="both")
dropped_implausible = (~plausible).sum()
df_clean = df_clean.loc[plausible]

with gzip.open(out_clean_path, "wt", encoding="utf-8") as f:
    df_clean.to_csv(f, index=False)

# --- “missing” and “other” unique values ---
def value_counts_series(frame):
    # Normalize whitespace for prettier grouping, but keep original strings for inspection
    tmp = frame["value"].fillna("").astype(str).str.strip()
    return tmp.value_counts(dropna=False)

missing_values_vc = value_counts_series(df_norange.loc[is_missing])
other_values_vc = value_counts_series(df_norange.loc[is_other])

# Save “missing” and “other” inventories to TSV for easy grepping
missing_values_vc.rename_axis("value").reset_index(name="count").to_csv(
    out_missing_values_tsv, sep="\t", index=False
)
other_values_vc.rename_axis("value").reset_index(name="count").to_csv(
    out_other_values_tsv, sep="\t", index=False
)

# --- summary report (JSON) ---
report = {
    "input_rows": int(len(df)),
    "dropped_ranges": int(is_range.sum()),
    "converted_numeric_rows": int(is_num.sum()),
    "clean_rows_after_plausibility": int(len(df_clean)),
    "dropped_implausible_numeric": int(dropped_implausible),
    "unit_class_counts": df["unit_class"].str.lower().value_counts().to_dict(),
    "missing_unique_values": int(missing_values_vc.shape[0]),
    "other_unique_values": int(other_values_vc.shape[0]),
    "missing_top20": missing_values_vc.head(20).to_dict(),
    "other_top20": other_values_vc.head(20).to_dict(),
    "outputs": {
        "clean_csv_gz": str(out_clean_path),
        "missing_values_tsv": str(out_missing_values_tsv),
        "other_values_tsv": str(out_other_values_tsv),
    },
}

# If group exists, add group counts to report
if "group" in df.columns:
    report["group_counts"] = df["group"].value_counts().to_dict()

with open(out_report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# --- console summary ---
print("=== Depth post-processing summary ===")
print(f"Input rows:                       {report['input_rows']:,}")
print(f"Dropped ranges:                   {report['dropped_ranges']:,}")
print(f"Converted to numeric (pre-filter) {report['converted_numeric_rows']:,}")
print(f"Dropped implausible numeric:      {report['dropped_implausible_numeric']:,}")
print(f"Clean rows (depth_m):             {report['clean_rows_after_plausibility']:,}")
print("\nUnit class counts:")
for k, v in report["unit_class_counts"].items():
    print(f"  {k:>20s}: {v:,}")

if "group_counts" in report:
    print("\nCounts by group (from input):")
    for k, v in report["group_counts"].items():
        print(f"  {k:>20s}: {v:,}")

print("\nTop 10 'missing' raw values:")
for v, c in list(report["missing_top20"].items())[:10]:
    print(f"  {c:>8}  {v}")

print("\nTop 100 'other' raw values:")
for v, c in list(report["other_top20"].items())[:100]:
    print(f"  {c:>8}  {v}")

print(f"\n✅ Clean CSV: {out_clean_path}")
print(f"📝 Report JSON: {out_report_path}")
print(f"🧭 Missing inventory: {out_missing_values_tsv}")
print(f"🧭 Other inventory:   {out_other_values_tsv}")





###############################################################################



# =============================================================================
# # # What happens in PART 3: 
# 
# # input: depth_clean.csv.gz (from PART 2)
# 
# 
# # steps: 
# #     - loads data
# #     - prints basic info of dataframe
# #     - summarize depth distribution
# #     - fieled and group (class) stats 
# #     - spot extreme values (displays the 5 shallowest and 5 deepest depths)
# =============================================================================
    

    
    
# PART 3 


# --- paths ---
path = Path("/Users/dgaio/MicrobeAtlasProject/depth_clean.csv.gz")

# --- load ---
df = pd.read_csv(path)

print(df.shape)
print(df.head())

print("\nDescriptive stats (depth_m):")
print(df["depth_m"].describe())

# quick counts by key
print("\nMost common raw keys:")
print(df["raw_key"].value_counts().head(15))

# if available, counts by group
if "group" in df.columns:
    print("\nCounts by group:")
    print(df["group"].value_counts())

# quantiles to understand distribution
print("\nDepth quantiles (overall):")
print(df["depth_m"].quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]))

if "group" in df.columns:
    print("\nDepth quantiles by group:")
    for g, sub in df.groupby("group"):
        q = sub["depth_m"].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        print(f"\nGroup: {g}")
        print(q)

# check min / max to spot weird cases
print("\nShallowest depths:")
print(df.sort_values("depth_m").head(5)[["sample_id","depth_m","raw_key","value","unit_class"]])

print("\nDeepest depths:")
print(df.sort_values("depth_m", ascending=False).head(5)[["sample_id","depth_m","raw_key","value","unit_class"]])


###############################################################################


# =============================================================================
# # # What happens in STEP 4: 
# 
# # input: depth_clean.csv.gz
# 
# # steps: 
# #     - defines depth bins 
# #     - plots 
#     
# # outputs: 
# #     - depth_distribution_overall.pdf
# #     - depth_distribution_by_group.pdf
# =============================================================================


# PART 4 — depth distribution plots (with PDF export)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- load cleaned depth (meters) ---
path = Path("/Users/dgaio/MicrobeAtlasProject/depth_clean.csv.gz")
df = pd.read_csv(path)
depth = df["depth_m"].astype(float)

# --- define bins ---
bins = [
    0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
    1000, 2500, 5000, 10000, 12001
]
labels = [
    "0–0.01","0.01–0.1","0.1–0.5","0.5–1","1–2","2–5","5–10","10–20",
    "20–50","50–100","100–200","200–500","0.5–1k","1–2.5k",
    "2.5–5k","5–10k","10–12k"
]

# --- binning ---
cats = pd.cut(depth, bins=bins, labels=labels, include_lowest=True, right=False)
bin_counts = cats.value_counts().sort_index()

x = np.arange(len(bin_counts))
y = bin_counts.values
xticklabels = bin_counts.index.astype(str)

# --- y-axis break settings ---
break_low = 50_000
break_high = 100_000   # explicit break range

# ---------- FIGURE 1: overall depth distribution ----------
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(12, 7),
    gridspec_kw={'height_ratios':[1, 3]}
)

# plot bars on both axes
ax_top.bar(x, y, color="teal")
ax_bottom.bar(x, y, color="teal")

# y-limits for the break
ax_top.set_ylim(break_high, y.max() * 1.05)
ax_bottom.set_ylim(0, break_low)

# remove connecting spines
ax_top.spines.bottom.set_visible(False)
ax_bottom.spines.top.set_visible(False)
ax_top.tick_params(labeltop=False)
ax_bottom.xaxis.tick_bottom()

# diagonal slashes
d = 0.015
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# annotate first bin
first_bin_count = y[0]
pct_first_bin = first_bin_count / y.sum() * 100 if y.sum() else 0.0
ax_top.text(
    x[0], y[0] + (y.max() * 0.02),
    f"{first_bin_count:,} ({pct_first_bin:.1f}%)",
    ha="center", va="bottom", fontsize=11, fontweight="bold"
)

# labels & title
ax_bottom.set_xticks(x)
ax_bottom.set_xticklabels(xticklabels, rotation=45, ha="right")
ax_bottom.set_ylabel("Sample count")
ax_bottom.set_xlabel("Depth (meters)")
fig.suptitle(
    "Depth distribution (meters)\n"
    f"Broken y-axis between 50k and 100k counts — first bin: {first_bin_count:,} samples ({pct_first_bin:.1f}%)",
    y=0.96
)

plt.tight_layout()

# --- save FIGURE 1 as PDF ---
out1 = path.parent / "depth_distribution_overall.pdf"
fig.savefig(out1, bbox_inches="tight")
print(f"[ok] Saved overall depth distribution → {out1}")

plt.show()

# ---------- FIGURE 2: per-group distribution ----------
if "group" in df.columns:
    groups = df["group"].unique().tolist()
    dist = {}
    for g in groups:
        dfg = df.loc[df["group"] == g, "depth_m"].astype(float)
        cg = pd.cut(dfg, bins=bins, labels=labels, include_lowest=True, right=False)
        vc = cg.value_counts().sort_index()
        dist[g] = (vc / vc.sum() * 100).fillna(0)

    dist_df = pd.DataFrame(dist, index=labels).fillna(0)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(dist_df))
    for g in dist_df.columns:
        ax2.bar(dist_df.index, dist_df[g].values, bottom=bottom, label=g)
        bottom += dist_df[g].values

    ax2.set_ylabel("Share within bin (%)")
    ax2.set_xlabel("Depth (meters)")
    ax2.set_xticklabels(dist_df.index, rotation=45, ha="right")
    ax2.set_title("Depth distribution by group (within-bin %)")
    ax2.legend(title="Group")
    plt.tight_layout()

    # --- save FIGURE 2 as PDF ---
    out2 = path.parent / "depth_distribution_by_group.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"[ok] Saved per-group depth distribution → {out2}")

    plt.show()

