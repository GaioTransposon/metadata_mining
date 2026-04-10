#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 21:48:34 2025

@author: dgaio
"""


# =============================================================================
# # # What happens in plot_altitude_map.py: 
#     
# # inputs (as arguments):
# #     - Christian's coordinates file (sample.coordinates.reparsed.filtered)
# #     - clean altitude file (altitude_clean.csv.gz)
# #     - gpt files from production (optional)
# #     - decide radius for dot size on map 
#     
# # steps: 
# #     - load files
# #     - normalizes country names (from GPT)
# #     - merges sample ids in common between Christian's coordinates and altitude file
# #     - merges GPT data
# #     - Bins altitudes into 15 ranges (e.g., *0–1 m*, *1–5 m*, …, *≥10k m*)
# #     - creates folium map
# #     - console prints diagnostics (rows merged, altitude ranges, high-altitude outliers)
#     
#     
# # output: 
# #     - altitude_map.html
# =============================================================================





# runs as: 
    
# python plot_altitude_map.py \
#   --work_dir ~/MicrobeAtlasProject \
#   --coordinates_file sample.coordinates.reparsed.filtered \
#   --altitude_file altitude_clean.csv.gz \
#   --output_map altitude_map.html \
#   --radius 3 \
#   --include_gpt \
#   --gpt_glob "production/gpt_clean_output*.csv"



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import folium
from folium import Map, LayerControl, FeatureGroup
from folium.features import Tooltip

# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Plot altitude samples on a world map, color-coded by altitude bins."
)
parser.add_argument("--work_dir", default=str(Path("/Users/dgaio/MicrobeAtlasProject")))
parser.add_argument("--coordinates_file", default="sample.coordinates.reparsed.filtered",
                    help="Space-delimited file with: label sample_id latitude longitude")
parser.add_argument("--altitude_file", default="altitude_clean.csv.gz",
                    help="CSV with at least: sample_id, altitude_m, value")
parser.add_argument("--output_map", default="altitude_map.html")

# Fixed radius
parser.add_argument("--radius", type=float, default=4.0,
                    help="Fixed marker radius in pixels (default: 4.0)")

# GPT options (optional)
parser.add_argument("--include_gpt", action="store_true",
                    help="If set, merge GPT geo/biome info into tooltips.")
parser.add_argument("--gpt_glob", default="production/gpt_clean_output*.csv",
                    help="Glob for GPT CSVs (expects columns: sample_id, geo_location, biome_label, sub_biome).")

args = parser.parse_args()

def abspath(base: str, path: str) -> str:
    if path is None: return None
    if os.path.isabs(path): return path
    return os.path.abspath(os.path.join(base, path))

work_dir = abspath(os.getcwd(), args.work_dir)
coordinates_path = abspath(work_dir, args.coordinates_file)
altitude_path = abspath(work_dir, args.altitude_file)
output_map_path = abspath(work_dir, args.output_map)

print(f"[info] work_dir:           {work_dir}")
print(f"[info] coordinates_file:   {coordinates_path}")
print(f"[info] altitude_file:      {altitude_path}")
print(f"[info] output_map:         {output_map_path}")
print(f"[info] include_gpt:        {args.include_gpt}")
print(f"[info] radius (px):        {args.radius}")

# ---------- Load coordinates ----------
coords = pd.read_csv(
    coordinates_path,
    delimiter=" ",
    header=None,
    names=["label", "sample_id", "latitude", "longitude"],
    na_values="None",
).drop(columns=["label"])
coords["latitude"] = pd.to_numeric(coords["latitude"], errors="coerce")
coords["longitude"] = pd.to_numeric(coords["longitude"], errors="coerce")
coords = coords.replace([np.inf, -np.inf], np.nan).dropna(subset=["latitude", "longitude"])

# ---------- Load altitudes ----------
alt = pd.read_csv(altitude_path)
alt = alt[["sample_id", "altitude_m", "value"]].dropna(subset=["altitude_m"]).drop_duplicates("sample_id", keep="first")

# ---------- Load GPT (optional) ----------
gpt_geo = pd.DataFrame(columns=["sample_id", "gpt_name", "biome_label", "sub_biome"])
if args.include_gpt:
    gpt_glob = os.path.join(work_dir, args.gpt_glob)
    gpt_files = glob.glob(gpt_glob)
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
            if "biome_label" not in df.columns:
                df["biome_label"] = pd.NA
            if "sub_biome" not in df.columns:
                df["sub_biome"] = pd.NA
            return df[["sample_id","gpt_name","biome_label","sub_biome"]]
        parts = [_read_gpt_file(f) for f in gpt_files]
        gpt_geo = pd.concat(parts, ignore_index=True).drop_duplicates("sample_id", keep="first")
        print(f"[info] GPT files merged: {len(gpt_files)}")
    else:
        print("[info] No GPT files matched; proceeding without GPT tooltips.")

# ---------- Merge ----------
df = coords.merge(alt, on="sample_id", how="inner")
if args.include_gpt and not gpt_geo.empty:
    df = df.merge(gpt_geo, on="sample_id", how="left")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["latitude", "longitude", "altitude_m"])

if df.empty:
    raise SystemExit("[error] No overlapping samples between coordinates and altitude data.")

# ---------- Diagnostics ----------
alt_all = pd.read_csv(altitude_path)[["sample_id", "altitude_m"]]
print(f"[diag] altitude_clean rows: {len(alt_all):,}")
print(f"[diag] altitude_clean max:  {alt_all['altitude_m'].max():.1f} m")
print(f"[diag] merged rows (with coords): {len(df):,}")
print(f"[diag] merged max altitude:       {df['altitude_m'].max():.1f} m")
def cnt_over(th):
    return (alt_all["altitude_m"] > th).sum(), (df["altitude_m"] > th).sum()
for th in (3000, 4000, 6000, 8000):
    a, b = cnt_over(th)
    print(f"[diag] >{th} m: clean={a:,}   merged={b:,}")
print(f"[info] merged rows (with altitude): {len(df):,}")

# ---------- Altitude bins ----------
bin_edges = [0, 1, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 4000, 6000, 8000, 10000, float('inf')]
bin_labels = ["0–1","1–5","5–10","10–20","20–50","50–100","100–200","200–500",
              "0.5–1k","1–2k","2–4k","4–6k","6–8k","8–10k","≥10k"]
df["alt_bin"] = pd.cut(df["altitude_m"], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False)

bin_colors = [
    "#440154", "#482878", "#3E4989", "#31688E", "#26828E",
    "#1F9E89", "#35B779", "#6DCD59", "#B4DE2C",
    "#FDE725", "#FCD225", "#FCA636", "#F57C4C", "#E75263", "#D43D4F"
]
color_map = dict(zip(bin_labels, bin_colors))

# ---------- Map ----------
center_lat = float(df["latitude"].mean())
center_lon = float(df["longitude"].mean())
m = Map(location=[center_lat, center_lon], zoom_start=2, tiles="CartoDB positron")

fg = FeatureGroup(name=f"Samples with altitude (n={len(df):,})", show=True)
fixed_radius = float(args.radius)

# build tooltips
include_gpt = args.include_gpt and ("gpt_name" in df.columns)
for _, row in df.iterrows():
    lat, lon = float(row["latitude"]), float(row["longitude"])
    sid = str(row["sample_id"])
    alt_m = float(row["altitude_m"])
    raw = str(row["value"])
    alt_bin = str(row["alt_bin"])
    color = color_map.get(alt_bin, "#000000")

    lines = [
        f"<b>sample_id:</b> {sid}",
        f"<b>lat, lon:</b> {lat:.4f}, {lon:.4f}",
        f"<b>altitude_m:</b> {alt_m:.1f} m",
        f"<b>altitude bin:</b> {alt_bin}",
        f"<b>raw value:</b> {raw}",
    ]
    if include_gpt:
        gname = row.get("gpt_name", pd.NA)
        biome = row.get("biome_label", pd.NA)
        subb  = row.get("sub_biome", pd.NA)
        if pd.notna(gname): lines.append(f"<b>geo_location:</b> {gname}")
        if pd.notna(biome): lines.append(f"<b>biome:</b> {biome}")
        if pd.notna(subb):  lines.append(f"<b>sub_biome:</b> {subb}")

    folium.CircleMarker(
        location=(lat, lon),
        radius=fixed_radius,
        color=color,
        fill=True,
        fill_opacity=0.75,
        weight=0,
        tooltip=Tooltip("<br>".join(lines), sticky=False),
    ).add_to(fg)

fg.add_to(m)

# ---------- Legend ----------
legend_items = "".join(
    f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
    f'<span style="display:inline-block;width:14px;height:14px;background:{color_map[l]};'
    f'margin-right:8px;border:1px solid #333;border-radius:2px;"></span>{l}</div>'
    for l in bin_labels
)
legend_html = f"""
<div style="
  position: fixed;
  bottom: 40px;
  left: 40px;
  z-index: 9999;
  background: rgba(255,255,255,0.92);
  padding: 10px 12px;
  border: 1px solid #777;
  border-radius: 8px;
  box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
  font-size: 12px;
  line-height: 1.1;
  ">
  <div style="font-weight:600; margin-bottom:6px;">Altitude (m) bins</div>
  {legend_items}
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

LayerControl(collapsed=False).add_to(m)
m.save(output_map_path)
print(f"[ok] Saved altitude map → {output_map_path}")



