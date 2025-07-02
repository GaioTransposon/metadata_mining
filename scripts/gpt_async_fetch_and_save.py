#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:07:58 2024

@author: dgaio
"""

# run as: 
# python github/metadata_mining/scripts/gpt_async_fetch_and_save.py

from __future__ import annotations


"""Fetch completed OpenAI batch-job outputs and save them to CSV.

✔ Works with openai-python ≥ 1.0
✔ Skips jobs whose output files are older than 30 days
✔ Avoids re-downloading CSVs that are already present
✔ Logs failed/expired jobs for manual re-submission if needed
"""


import csv
import glob
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set

from openai import NotFoundError, OpenAI  # SDK ≥ 1.0
import argparse


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
cli = argparse.ArgumentParser(
    description="Fetch completed OpenAI batch outputs and save tidy CSVs."
)
cli.add_argument(
    "--work_dir",
    default=".",
    help="Base working directory (default = current dir; '/MicrobeAtlasProject' in Docker)",
)
cli.add_argument(
    "--api_key_path",
    required=True,
    help="OpenAI API-key filename (relative to work_dir)",
)
args = cli.parse_args()

WORK_DIR       = Path(os.path.abspath(args.work_dir))
API_KEY_FILE   = WORK_DIR / args.api_key_path
FAILED_SAMPLES = WORK_DIR / "failed_async_samples.txt"
FAILED_SAMPLES.parent.mkdir(parents=True, exist_ok=True)




# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────


def init_openai_client(api_key_path: str) -> OpenAI:
    """Return a configured OpenAI client instance."""
    with open(api_key_path, "r", encoding="utf-8") as fh:
        api_key = fh.read().strip()
    return OpenAI(api_key=api_key)


def retrieve_results(client: OpenAI, batch_job_id: str) -> str | None:
    batch_job = client.batches.retrieve(batch_job_id)
    print("\n", batch_job, "\n")

    # Skip if output is older than 30 days
    if batch_job.completed_at and (time.time() - batch_job.completed_at > 30 * 24 * 3600):
        print(f"{batch_job_id} → completed more than 30 days ago — skipping.")
        return None

    # Handle all-complete failure batches
    if batch_job.status == "completed":
        if batch_job.output_file_id:
            file_content = client.files.content(batch_job.output_file_id)
            return file_content.text
        elif batch_job.error_file_id:
            error_file = client.files.content(batch_job.error_file_id)
            error_text = error_file.text
            print(f"{batch_job_id} → completed with errors only.")
            
            error_log_path = WORK_DIR / f"{batch_job_id}_error.jsonl"
            error_log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(error_log_path, "w") as f:
                f.write(error_text)
            print(f"Saved error file to {error_log_path}")
            return None
        else:
            print(f"{batch_job_id} → completed but has no output or error file.")
            return None

    print(f"{batch_job_id} → still in progress.")
    return None



# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────────────


def convert_jsonl_content_to_csv(jsonl_content: str, output_csv_path: Path, failed_samples_path: Path) -> None:
    """Convert *JSON* batch responses to a tidy CSV."""
    lines: List[str] = jsonl_content.splitlines()

    with open(output_csv_path, "w", newline="") as csv_file, open(failed_samples_path, "a") as failed_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_id", "biome_label", "geo_location", "keywords", "sub_biome"])

        for line in lines:
            try:
                obj = json.loads(line)
                content = json.loads(obj["response"]["body"]["choices"][0]["message"]["content"])
                writer.writerow(
                    [
                        content.get("sample-id", "N/A"),
                        content.get("biome-label", "N/A"),
                        content.get("geo-location", "N/A"),
                        content.get("keywords", "N/A"),
                        content.get("sub-biome", "N/A"),
                    ]
                )
            except Exception as exc:  # Broad catch is OK for batch post-mortem
                failed_file.write(f"Failed to process line: {line}\nError: {exc}\n")


INLINE_SAMPLE_RE = re.compile(r"(ERS|SRS|DRS)\d+(_{2,4}).*")


def parse_inline_responses(lines: Iterable[str], output_csv_path: Path) -> None:
    """Parse *inline* responses produced by your custom prompt."""
    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_id", "biome_label", "geo_location", "keywords", "sub_biome"])

        for line in lines:
            match = INLINE_SAMPLE_RE.search(line)
            if match:
                parts = re.split(r"_{2,4}", match.group())
                writer.writerow(parts)
            else:
                print(f"Failed to parse line: {line}")


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_existing_batch_ids(directory: Path) -> Set[str]:
    pattern = directory / "gpt_clean_output*batch*.csv"
    return {
        "batch_" + Path(p).stem.split("_batch")[-1].split("_dt")[0]
        for p in glob.glob(str(pattern))
    }


def log_failed_batch(directory: Path, batch_job_id: str) -> None:
    with open(directory / "failed_async_batches.txt", "a") as fh:
        fh.write(batch_job_id + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main script logic
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: C901 – a little long but readable
    project_dir        = WORK_DIR
    api_key_file       = API_KEY_FILE
    failed_samples_path = FAILED_SAMPLES


    print(api_key_file)
    client = init_openai_client(str(api_key_file))

    existing_csvs = get_existing_batch_ids(project_dir)
    print(existing_csvs)

    # Read batch-job metadata recorded at submission time
    with open(project_dir / "batch_job_info.json", "r") as fh:
        batch_info_list = json.load(fh)

    for info in batch_info_list:
        batch_id = info["batch_job_id"]
        fmt = info["output_format"]

        print("batch_job_id", batch_id)

        if batch_id in existing_csvs:
            print(f"CSV for {batch_id} already exists – skipping.")
            continue

        content = retrieve_results(client, batch_id)
        if not content:
            log_failed_batch(project_dir, batch_id)
            continue

        # Build output filename
        out_csv = (
            project_dir
            / (
                f"gpt_clean_output_nspb{info['nspb']}_chunking{info['chunking']}"
                f"_chunksize{info['chunksize']}_model{info['model']}_temp{info['temperature']}"
                f"_maxtokens{info['max_tokens']}_topp{info['top_p']}_freqp{info['frequency_penalty']}"
                f"_presp{info['presence_penalty']}_rs{info['rs']}_format{fmt}"
                f"_batch{batch_id.split('_')[-1]}_dt{info['datetime']}.csv"
            )
        )

        try:
            if fmt == "json":
                convert_jsonl_content_to_csv(content, out_csv, failed_samples_path)
            elif fmt == "inline":
                lines: List[str] = [
                    json.loads(line)["response"]["body"]["choices"][0]["message"]["content"]
                    for line in content.splitlines()
                    if line.strip()
                ]
                parse_inline_responses(lines, out_csv)
            print("Batch completed – results saved to", out_csv)
        except Exception as exc:  # log and continue with next batch
            print(f"Error processing batch {batch_id}: {exc}")
            log_failed_batch(project_dir, batch_id)


if __name__ == "__main__":
    main()








