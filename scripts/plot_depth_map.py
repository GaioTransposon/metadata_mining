#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 18:26:32 2025

@author: dgaio
"""




# =============================================================================
# # # What happens in plot_depth_map.py: 
#     
# # inputs (as arguments):
# #     - Christian's coordinates file (sample.coordinates.reparsed.filtered)
# #     - clean depth file (depth_clean.csv.gz)
# #     - gpt files from production (optional with --gpt_include)
# #     - decide radius for dot size on map 
# #     - decide per_map_cap: maximum number of samples to show in the map
#     
# # steps: 
# #     - load files
# #     - normalizes country names (from GPT)
# #     - merges sample ids in common between Christian's coordinates and depth file
# #     - merges GPT data
# #     - picks random samples (uses seed)
# #     - Bins depths into ranges (from 0–0.01 m up to ≥10k m)
# #     - creates folium map
# #     - console prints diagnostics (rows merged, depth ranges, high-depth outliers)
#     
#     
# # output: 
# #     - depth_map*.html
# =============================================================================







# run as: 
    
# python plot_depth_map.py \
#   --work_dir ~/MicrobeAtlasProject \
#   --coordinates_file sample.coordinates.reparsed.filtered \
#   --depth_file depth_clean.csv.gz \
#   --output_map depth_map.html \
#   --per_map_cap 80000 \
#   --radius 5 \
#   --include_gpt \
#   --gpt_glob "production/gpt_clean_output*.csv"




import argparse, os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium import Map, LayerControl, FeatureGroup
from folium.features import Tooltip

# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Plot ALL depth samples as multiple HTML maps (fixed-size markers, ≤ per_map_cap each)."
)
parser.add_argument("--work_dir", default=str(Path("/Users/dgaio/MicrobeAtlasProject")))
parser.add_argument("--coordinates_file", default="sample.coordinates.reparsed.filtered",
                    help="Space-delimited file with: label sample_id latitude longitude")
parser.add_argument("--depth_file", default="depth_clean.csv.gz",
                    help="CSV with at least: sample_id, depth_m, value (optional: group)")
parser.add_argument("--output_map", default="depth_map.html",
                    help="Base filename; script writes *_part1.html, *_part2.html, ...")
parser.add_argument("--per_map_cap", type=int, default=80000,
                    help="Max points per map (default 80k).")
parser.add_argument("--radius", type=float, default=5.0,
                    help="Fixed CircleMarker radius (pixels).")

# GPT options (optional)
parser.add_argument("--include_gpt", action="store_true",
                    help="Merge GPT geo/biome info into tooltips.")
parser.add_argument("--gpt_glob", default="production/gpt_clean_output*.csv",
                    help="Glob for GPT CSVs (expects columns: sample_id, geo_location, biome_label, sub_biome).")

args = parser.parse_args()

def abspath(base: str, path: str) -> str:
    if path is None: return None
    if os.path.isabs(path): return path
    return os.path.abspath(os.path.join(base, path))

work_dir = abspath(os.getcwd(), args.work_dir)
coordinates_path = abspath(work_dir, args.coordinates_file)
depth_path = abspath(work_dir, args.depth_file)
output_map_base = abspath(work_dir, args.output_map)

print(f"[info] work_dir:         {work_dir}")
print(f"[info] coordinates_file: {coordinates_path}")
print(f"[info] depth_file:       {depth_path}")
print(f"[info] output_map base:  {output_map_base}")
print(f"[info] per_map_cap:      {args.per_map_cap:,}")
print(f"[info] fixed radius:     {args.radius}")
print(f"[info] include_gpt:      {args.include_gpt}")

# ---------- Load coordinates ----------
coords = pd.read_csv(
    coordinates_path,
    delimiter=" ",
    header=None,
    names=["label", "sample_id", "latitude", "longitude"],
    na_values=["None", "", "NA", "nan"]
).drop(columns=["label"])
coords["latitude"]  = pd.to_numeric(coords["latitude"], errors="coerce")
coords["longitude"] = pd.to_numeric(coords["longitude"], errors="coerce")
coords = coords.dropna(subset=["latitude", "longitude"])
coords = coords[(coords["latitude"].between(-90, 90)) & (coords["longitude"].between(-180, 180))]
coords.columns = [c.strip() for c in coords.columns]
print(f"[diag] coords usable: {len(coords):,}")

# ---------- Load depth ----------
dep = pd.read_csv(depth_path)
dep.columns = [c.strip() for c in dep.columns]
need_cols = {"sample_id", "depth_m"}
missing = need_cols - set(dep.columns)
if missing:
    raise SystemExit(f"[error] depth file missing required columns: {missing}")

keep_cols = ["sample_id", "depth_m", "value"]
if "group" in dep.columns:
    keep_cols.append("group")
dep = dep[keep_cols].copy()
dep["depth_m"] = pd.to_numeric(dep["depth_m"], errors="coerce")
dep = dep.dropna(subset=["depth_m"]).drop_duplicates("sample_id", keep="first")
print(f"[diag] depth usable: {len(dep):,}")

# ---------- Load GPT (optional) ----------
gpt_geo = pd.DataFrame(columns=["sample_id", "gpt_name", "biome_label", "sub_biome"])
if args.include_gpt:
    gpt_files = glob.glob(os.path.join(work_dir, args.gpt_glob))
    if gpt_files:
        def _read_gpt_file(fpath: str) -> pd.DataFrame:
            df = pd.read_csv(fpath)
            if "sample_id" not in df.columns:
                return pd.DataFrame(columns=["sample_id", "gpt_name", "biome_label", "sub_biome"])
            cols = ["sample_id"] + [c for c in ["geo_location","biome_label","sub_biome"] if c in df.columns]
            df = df[cols].copy()
            if "geo_location" in df.columns:
                df["geo_location"] = (
                    df["geo_location"].astype(str)
                        .str.replace(":", " ", regex=False)
                        .str.replace("US", "United States", regex=False)
                        .str.replace("USA", "United States of America", regex=False)
                        .str.replace("Viet Nam", "Vietnam", regex=False)
                        .str.replace("Czech Republic", "Czechia", regex=False)
                )
                df.rename(columns={"geo_location": "gpt_name"}, inplace=True)
            for c in ["biome_label", "sub_biome"]:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[["sample_id","gpt_name","biome_label","sub_biome"]]
        gpt_geo = pd.concat([_read_gpt_file(f) for f in gpt_files], ignore_index=True)\
                   .drop_duplicates("sample_id", keep="first")
        print(f"[info] GPT files merged: {len(gpt_files)}")
    else:
        print("[info] No GPT files matched; proceeding without GPT tooltips.")

# ---------- Merge ----------
df = coords.merge(dep, on="sample_id", how="inner")
if args.include_gpt and not gpt_geo.empty:
    df = df.merge(gpt_geo, on="sample_id", how="left")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["latitude", "longitude", "depth_m"])
df.columns = [c.strip() for c in df.columns]
total_points = len(df)
print(f"[diag] merged (coords×depth): {total_points:,}")

if df.empty:
    raise SystemExit("[error] No overlapping samples between coordinates and depth data.")

# ---------- Depth bins (meters) ----------
bin_edges = [
    0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
    1000, 2500, 5000, 10000, float('inf')
]
bin_labels = [
    "0–0.01","0.01–0.1","0.1–0.5","0.5–1","1–2","2–5","5–10","10–20",
    "20–50","50–100","100–200","200–500","0.5–1k","1–2.5k",
    "2.5–5k","5–10k","≥10k"
]
bin_colors = [
    "#440154", "#482878", "#3E4989", "#31688E", "#26828E",
    "#1F9E89", "#35B779", "#6DCD59", "#B4DE2C",
    "#FDE725", "#FCD225", "#FCA636", "#F57C4C", "#E75263",
    "#D43D4F", "#B8252A", "#7F0000"
]
color_map = dict(zip(bin_labels, bin_colors))

df["depth_bin"] = pd.cut(
    df["depth_m"].astype(float),
    bins=bin_edges, labels=bin_labels,
    include_lowest=True, right=False
)
print("[diag] bin counts:")
print(df["depth_bin"].value_counts().reindex(bin_labels, fill_value=0).astype(int))

# ---------- Split ALL points into multiple maps ----------
cap = max(1, int(args.per_map_cap))
n_maps = math.ceil(total_points / cap)

base = Path(output_map_base)
parent, stem, suffix = base.parent, base.stem, (base.suffix or ".html")
print(f"[info] creating {n_maps} maps (≤ {cap:,} pts each) → {parent}")

# deterministic order: sort by sample_id
rng = np.random.default_rng(42)  # fixed seed for reproducibility
df_shuffled = df.sample(frac=1, random_state=rng.integers(0, 1e9)).reset_index(drop=True)
chunks = np.array_split(df_shuffled, n_maps)

for i, dfi in enumerate(chunks, start=1):
    out_path = parent / f"{stem}_part{i}{suffix}"
    print(f"[info]   part {i}/{n_maps}: {len(dfi):,} samples → {out_path}")

    center_lat = float(dfi["latitude"].mean())
    center_lon = float(dfi["longitude"].mean())
    m = Map(location=[center_lat, center_lon], zoom_start=2, tiles="CartoDB positron")

    fg = FeatureGroup(name=f"Depth samples (part {i}/{n_maps}, n={len(dfi):,})", show=True)
    R = float(args.radius)

    for _, row in dfi.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        depth_m = float(row["depth_m"])
        dbin = row.get("depth_bin")
        if pd.isna(dbin):
            dbin = pd.cut(pd.Series([depth_m]), bins=bin_edges, labels=bin_labels,
                          include_lowest=True, right=False).iloc[0]
        depth_bin = str(dbin)
        color = color_map.get(depth_bin, "#000000")

        lines = [
            f"<b>sample_id:</b> {row['sample_id']}",
            f"<b>lat, lon:</b> {lat:.4f}, {lon:.4f}",
            f"<b>depth_m:</b> {depth_m:.2f} m",
            f"<b>depth bin:</b> {depth_bin}",
        ]
        raw = row.get("value", "")
        if isinstance(raw, str) and raw.strip():
            lines.append(f"<b>raw value:</b> {raw}")
        if "group" in row and pd.notna(row["group"]):
            lines.append(f"<b>group:</b> {row['group']}")
        if args.include_gpt:
            gname = row.get("gpt_name", pd.NA)
            biome = row.get("biome_label", pd.NA)
            subb  = row.get("sub_biome", pd.NA)
            if pd.notna(gname): lines.append(f"<b>geo_location:</b> {gname}")
            if pd.notna(biome): lines.append(f"<b>biome:</b> {biome}")
            if pd.notna(subb):  lines.append(f"<b>sub_biome:</b> {subb}")

        folium.CircleMarker(
            location=(lat, lon),
            radius=R,            # fixed size
            color=color,
            fill=True,
            fill_opacity=0.75,
            weight=0,
            tooltip=Tooltip("<br>".join(lines), sticky=False),
        ).add_to(fg)

    fg.add_to(m)

    # Legend
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<span style="display:inline-block;width:14px;height:14px;background:{color_map.get(l, "#000000")};'
        f'margin-right:8px;border:1px solid #333;border-radius:2px;"></span>{l}</div>'
        for l in bin_labels
    )
    legend_html = f"""
    <div style="
      position: fixed; bottom: 40px; left: 40px; z-index: 9999;
      background: rgba(255,255,255,0.92); padding: 10px 12px;
      border: 1px solid #777; border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
      font-size: 12px; line-height: 1.1;">
      <div style="font-weight:600; margin-bottom:6px;">Depth (m) bins</div>
      {legend_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))

print(f"[ok] Wrote {n_maps} maps (total points: {total_points:,}).")


