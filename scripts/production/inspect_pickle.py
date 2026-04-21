#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:54:00 2026

@author: dgaio
"""

import pickle
import sys
from collections import defaultdict


def inspect_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"\nTotal entries: {len(data)}\n")

    # Group by prefix
    groups = defaultdict(list)
    for k, v in data.items():
        prefix = k.split("_")[0]
        groups[prefix].append((k, v))

    # Sort keys within each group (important!)
    for prefix in groups:
        groups[prefix] = sorted(groups[prefix], key=lambda x: x[0])

    # Display per group
    for prefix, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"\n=== {prefix} ({len(items)} entries) ===\n")

        n = len(items)

        # First 100
        print("---- FIRST 100 ----")
        for k, v in items[:100]:
            print(f"{k} -> {v}")

        # Last 100 (avoid duplication if small)
        if n > 100:
            print("\n---- LAST 100 ----")
            for k, v in items[-100:]:
                print(f"{k} -> {v}")

        print("\n" + "=" * 50)

    print("\nDone.\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pickle.py <path_to_pickle>")
        sys.exit(1)

    inspect_pickle(sys.argv[1])