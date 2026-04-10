#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 19:34:32 2025

@author: dgaio
"""


import gzip
import io
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np



###############################################################################

# # What happens in PART 1: 
    
# =============================================================================
# # input: sample.info.gz
# 
# # steps: 
# #     - detect sample ids
# #     - catch all altitude -containing fields 
# #     - normalize fields
# #     - store values and tag them into classes: `missing`, `sea_level`, `masl`, `feet`, `km`, `cm`, `meters`, `bare_number`, `range_or_composite`, or `other`
# #     - count occurrences of raw + normalized fields 
# #     - ⚠️ All matching fields are kept i.e. duplicates per sample are preserved 
# #     - write all records to `altitude_data.csv.gz`
# #     - print stats to console
# 
# # output: altitude_data.csv.gz
# =============================================================================


# PART 1


gz_path = Path("/Users/dgaio/MicrobeAtlasProject/sample.info.gz")
out_path = gz_path.with_name("altitude_data.csv.gz")

# capture key=value lines
kv_re = re.compile(r'^\s*([^=\s]+)\s*=\s*(.*)$')
# block header lines like ">SRS123456", ">ERS676566", etc.
header_re = re.compile(r'^\s*>\s*([A-Za-z0-9_.:-]+)\s*$')

# keys that mention "altitude" somewhere in the key (case-insensitive)
alt_key_re = re.compile(r'altitude', re.IGNORECASE)

def normalize_key(k: str) -> str:
    k0 = k
    k = k.lower()
    k = re.sub(r'\([^)]*\)', '', k)                # remove (...) bits
    k = re.sub(r'^(sample|experiment)[_-]+', '', k) # drop common prefixes
    k = re.sub(r'[^a-z0-9]+', '_', k)               # unify separators
    k = re.sub(r'_+', '_', k).strip('_')            # collapse trim
    return k or k0.lower()

def classify_value_units(v: str) -> str:
    s = v.strip().lower()

    # common 'not available'
    if s in {
        '', 'na', 'n/a', 'null', 'none', 'unknown',
        'missing', 'not collected', 'not applicable'
    }:
        return 'missing'

    # explicit "sea level" (English or Spanish phrase)
    # if the entire string is essentially 'sea level' (optionally with punctuation/whitespace)
    if re.fullmatch(r'\s*sea\s+level\s*\.?,?\s*', s) or re.fullmatch(r'\s*nivel\s+del\s+mar\s*\.?,?\s*', s):
        return 'sea_level'

    # MASL variants (English + Spanish abbreviations)
    # e.g., "2300 masl", "2300 m a.s.l", "2300 amsl", "2300 msnm", "2300 msm"
    if (
        'masl' in s or 'm a.s.l' in s or 'm asl' in s or 'amsl' in s
        or re.search(r'\bmsnm+\b', s)  # msnm or msnmm
        or re.search(r'\bmsm\b', s)
        or 'nivel del mar' in s  # phrase appears alongside a number
    ):
        return 'masl'

    if re.search(r'\b(ft|feet)\b', s):
        return 'feet'
    if re.search(r'\b(km|kilometer|kilometre|kilometers|kilometres)\b', s):
        return 'km'
    if re.search(r'\b(cm|centimeter|centimetre|centimeters|centimetres)\b', s):
        return 'cm'
    # m/meters/metres OR bare 'm' after number (e.g., "1200 m")
    if re.search(r'\b(m|meter|metre|meters|metres)\b', s) or re.search(r'\d\s*m\b', s):
        return 'meters'
    # plain number (allow , or . as decimal/thousands)
    if re.fullmatch(r'\s*-?\d+(?:[.,]\d+)?\s*', s):
        return 'bare_number'
    # ranges like "100-200 m" or "100 to 200m"
    if re.search(r'-| to ', s):
        return 'range_or_composite'
    return 'other'


raw_key_counts = Counter()
norm_key_counts = Counter()
norm_to_examples = defaultdict(lambda: Counter())
unit_counts = Counter()

records = []
current_sample_id = None
unknown_sid_skipped = 0

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
        if not alt_key_re.search(key):
            continue

        raw_key_counts[key] += 1

        nk = normalize_key(key)
        norm_key_counts[nk] += 1
        if sum(norm_to_examples[nk].values()) < 20:
            norm_to_examples[nk][key] += 1

        unit_counts[classify_value_units(val)] += 1

        records.append({
            "sample_id": current_sample_id,
            "normalized_key": nk,
            "raw_key": key,
            "value": val.strip(),
            "unit_class": classify_value_units(val),
        })

# --- SAVE OUTPUT ---
with gzip.open(out_path, "wt", encoding="utf-8", newline="") as gz_out:
    writer = csv.DictWriter(gz_out, fieldnames=["sample_id", "normalized_key", "raw_key", "value", "unit_class"])
    writer.writeheader()
    writer.writerows(records)

# --- REPORT ---
print("Top raw keys (as-is):")
for k, n in raw_key_counts.most_common(100):
    print(f"{n}\t{k}")

print("\nTop normalized keys:")
for k, n in norm_key_counts.most_common(100):
    print(f"{n}\t{k}")

print("\nExamples per top normalized key (up to a few variants each):")
for nk, _ in norm_key_counts.most_common(100):
    ex = ', '.join([f"{k}×{c}" for k, c in norm_to_examples[nk].most_common(5)])
    print(f"- {nk}: {ex}")

print("\nValue/unit style histogram (very rough):")
for u, n in unit_counts.most_common():
    print(f"{n}\t{u}")

print(f"\n✅ Saved {len(records):,} altitude-related entries to {out_path}")


###############################################################################



# =============================================================================
# # What happens in PART 2: 
#     
# # input: altitude_data.csv.gz (from PART 1)
# 
# # steps: 
# #     - discard ranges or composites
# #     - parsing numbers (commas, dots)
# #     - equals sea_level to 0 m 
# #     - keeps meters, masl, bare_number as meters
# #     - converts feet, km, cm to meters
# #     - missing, other and range_or_composite become NA
# #     - plausibility filter: 0.0 ≤ altitude_m ≤ 10,000.0
# #     - prints stats to console and writes reports 
# #     - saves to files
#      
# # outputs: 
# #     - altitude_clean.csv.gz
# #     - altitude_missing_values.tsv
# #     - altitude_other_values.tsv
# #     - altitude_report.json
# =============================================================================



# PART 2


# --- paths ---
in_path = Path("/Users/dgaio/MicrobeAtlasProject/altitude_data.csv.gz")
out_clean_path = in_path.with_name("altitude_clean.csv.gz")
out_report_path = in_path.with_name("altitude_report.json")
out_missing_values_tsv = in_path.with_name("altitude_missing_values.tsv")
out_other_values_tsv = in_path.with_name("altitude_other_values.tsv")

# --- helpers ---

_num_re = re.compile(r"[+-]?\d[\d.,]*")



# what's happening in _parse_num() (examples)
# input: "1,234.56"	after step 1: "1,234.56" after step 2: 	"1234.56"	parsed as: 1234.56
# input: "12,000"	after step 1: "12000" after step 2: 	"12000"	parsed as: 12000.0
# input: "1.234,56"	after step 1: "1.234.56" after step 2: 	"1234.56"	parsed as: 1234.56
def _parse_num(s: str):
    """Extract first number from a string; handle decimal comma."""
    if not isinstance(s, str):
        return None
    m = _num_re.search(s)
    if not m:
        return None
    x = m.group(0)
    # step 1: if there is a comma but no dot, treat comma as decimal separator. 
    if ',' in x and '.' not in x:
        x = x.replace(',', '.')
    # step 2: remove any remaining thousands separators
    x = x.replace(',', '')
    try:
        return float(x)
    except Exception:
        return None


def to_meters(value_str: str, unit_class: str):
    """Return altitude in meters or None if not convertible."""
    unit_class = (unit_class or "").lower()

    # Map explicit sea-level tokens to 0 m
    if unit_class == "sea_level":
        return 0.0

    if unit_class in ("missing", "other", "range_or_composite"):
        return None

    num = _parse_num(value_str)
    if num is None:
        return None

    if unit_class in ("meters", "masl", "bare_number"):
        return num
    if unit_class == "feet":
        return num * 0.3048
    if unit_class == "km":
        return num * 1000.0
    if unit_class == "cm":
        return num / 100.0

    # --- Fallback: unexpected class → treat as 'other' ---
    print(f"[info] Unexpected unit_class='{unit_class}' for value='{value_str}' → reclassified as 'other'")
    return None




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

# --- convert to meters ---
df_norange["altitude_m"] = df_norange.apply(
    lambda r: to_meters(r["value"], r["unit_class"]), axis=1
)

# Keep rows with numeric altitude
is_num = df_norange["altitude_m"].notna()

# capture what's missing/other/unparsed for reporting
is_missing = df_norange["unit_class"].str.lower().eq("missing")
is_other = df_norange["unit_class"].str.lower().eq("other")
is_unparsed_but_expected = (~is_range) & (~is_num) & (~is_missing) & (~is_other)

# --- save cleaned dataset ---
clean_cols = ["sample_id", "altitude_m", "normalized_key", "raw_key", "value", "unit_class"]
df_clean = df_norange.loc[is_num, clean_cols].copy()

# --- plausibility filter (0 to 10,000 m) ---
# Negative altitudes (below sea level) are removed.
plausible = df_clean["altitude_m"].between(0.0, 10000.0, inclusive="both")
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

with open(out_report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# --- console summary ---
print("=== Altitude post-processing summary ===")
print(f"Input rows:                       {report['input_rows']:,}")
print(f"Dropped ranges:                   {report['dropped_ranges']:,}")
print(f"Converted to numeric (pre-filter) {report['converted_numeric_rows']:,}")
print(f"Dropped implausible numeric:      {report['dropped_implausible_numeric']:,}")
print(f"Clean rows (altitude_m):          {report['clean_rows_after_plausibility']:,}")
print("\nUnit class counts:")
for k, v in report["unit_class_counts"].items():
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
# # input: altitude_clean.csv.gz (from PART 2)
# 
# # steps: 
# #     - loads data
# #     - prints basic info of dataframe
# #     - summarize altitude distribution
# #     - inspect metadata consistency (10 most frequent raw_key fields)
# #     - spot extreme values (displays the 5 lowest and 5 highest altitude_m samples)
# =============================================================================

    


# PART 3


path = "/Users/dgaio/MicrobeAtlasProject/altitude_clean.csv.gz"
df = pd.read_csv(path)

print(df.shape)
print(df.head())

print("\nDescriptive stats:")
print(df["altitude_m"].describe())

# quick counts by key
print("\nMost common raw keys:")
print(df["raw_key"].value_counts().head(10))

# check min / max to spot weird cases
print("\nExtreme altitudes:")
print(df.sort_values("altitude_m").head(10))
print(df.sort_values("altitude_m", ascending=False).head(10))



###############################################################################



# =============================================================================
# # What happens in PART 4: 
#     
# # input: altitude_clean.csv.gz (from PART 2)
# 
# # steps: 
# #     - defines altitude bins
# #     - plots
# 
# # outputs: 
# #     - altitude_distribution.pdf
# =============================================================================



# PART 4


# --- load data ---
path = Path("/Users/dgaio/MicrobeAtlasProject/altitude_clean.csv.gz")
df = pd.read_csv(path)

# --- define bins ---
bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
labels = ["0–1","1–5","5–10","10–20","20–50","50–100","100–200","200–500",
          "0.5–1k","1–2k","2–4k","4–6k","6–8k","8–10k"]

cats = pd.cut(df["altitude_m"], bins=bins, labels=labels, include_lowest=True, right=False)
bin_counts = cats.value_counts().sort_index()

# --- convert to arrays ---
x = np.arange(len(bin_counts))
y = bin_counts.values
xticklabels = bin_counts.index.astype(str)

# --- define break range ---
break_low = 5_000
break_high = 50_000

# --- create figure ---
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(10,6),
    gridspec_kw={'height_ratios':[1, 3]}
)

# --- plot bars ---
ax_top.bar(x, y, color="teal")
ax_bottom.bar(x, y, color="teal")

# --- adjust y-limits for the break ---
ax_top.set_ylim(break_high, y.max() * 1.05)
ax_bottom.set_ylim(0, break_low)

# --- hide spines and add break marks ---
ax_top.spines.bottom.set_visible(False)
ax_bottom.spines.top.set_visible(False)
ax_top.tick_params(labeltop=False)
ax_bottom.xaxis.tick_bottom()

d = 0.015
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# --- annotate first bin ---
first_bin_count = y[0]
pct_first_bin = first_bin_count / y.sum() * 100
ax_top.text(
    x[0], y[0] + 1000,
    f"{pct_first_bin:.1f}%",
    ha="center", va="bottom", fontsize=11, fontweight="bold", color="black"
)

# --- labels and title ---
ax_bottom.set_xticks(x)
ax_bottom.set_xticklabels(xticklabels, rotation=45, ha="right")
ax_bottom.set_ylabel("Count")
fig.suptitle(f"Altitude bins (m) — y-axis break between 5k and 50k\n"
             f"First bin (0–1 m): {pct_first_bin:.1f}% of samples", y=0.96)

plt.tight_layout()

# --- save plots ---
out_pdf = path.with_name("altitude_distribution.pdf")
fig.savefig(out_pdf)
print(f"[ok] Saved plots to:\n  {out_pdf}")

# --- show ---
plt.show()




