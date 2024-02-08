#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:23:35 2024

@author: dgaio
"""

import os

def remove_samples(processed_samples_file, to_del_file, output_file):
    # Load the IDs to delete
    with open(to_del_file, 'r') as f:
        to_del = set(f.read().splitlines())

    # Process the original file and write out only those not in to_del
    with open(processed_samples_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            sample_id = line.strip()
            if sample_id not in to_del:
                out_f.write(line)

def main():
    work_dir = "/mnt/mnemo5/dgaio/MicrobeAtlasProject/"
    processed_samples_file = os.path.join(work_dir, 'processed_samples_file.txt')
    to_del_file = os.path.join(work_dir, 'to_del.txt')
    output_file = os.path.join(work_dir, 'processed_samples_file_updated.txt')

    remove_samples(processed_samples_file, to_del_file, output_file)

if __name__ == "__main__":
    main()
