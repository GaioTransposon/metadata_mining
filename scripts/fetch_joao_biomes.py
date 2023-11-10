#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:33:33 2023

@author: dgaio
"""

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import json 



def read_mapdb_h5(h5_path, otu_data=True, meta_data=True):
    """Example file: /mnt/mnemo3/janko/data/microbe_atlas/sruns-otus.97.otutable_plusMetaNoHeader_taxonomy_unmapped.h5"""
    f = h5py.File(h5_path, 'r')
    result_dict = {}

    # read otu data
    if otu_data:
        data_handle = f["otu_table"]
        col_ptr, nzvals, rowindices, otu_index, sample_index = [np.array(data_handle[sub_key]) for sub_key in ['data_colptr', 'data_nzval', 'data_rowval', 'oids', 'sids']]

        ## correct indexing (julia starts at 1)
        col_ptr -= 1
        rowindices -= 1

        otutable_sparse = csc_matrix((nzvals,rowindices,col_ptr), shape=(data_handle["m"][()],data_handle["n"][()]))
        result_dict["otu_data"] = {"otu_table": otutable_sparse, "otu_index": otu_index, "sample_index": sample_index}

    # read meta data
    if meta_data:
        meta_handle = f["meta_data"]
        result_dict["meta_data"] = {sub_key: pd.DataFrame(np.array(meta_handle[sub_key]).T) for sub_key in meta_handle.keys()}

    f.close()

    return result_dict


# Assuming `result` is the dictionary returned from your function
def save_results(result_dict):
    # Save the entire dictionary as JSON
    # Note: This will not include the sparse matrix in its original format
    with open(output, 'w') as f_json:
        # Convert numpy arrays to lists for JSON serialization
        json_safe_dict = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in result_dict.items()
        }
        json.dump(json_safe_dict, f_json, indent=4)

    # Save tabular data to CSV
    # Assuming meta_data is a DataFrame or can be converted to one
    if 'meta_data' in result_dict:
        for key, df in result_dict['meta_data'].items():
            df.to_csv(f'/mnt/data/meta_data_{key}.csv', index=False)



h5_path = "/mnt/mnemo3/janko/data/microbe_atlas/sruns-otus.97.otutable_plusMetaNoHeader_taxonomy_unmapped.h5"

output = "/mnt/mnemo5/dgaio/joao_biomes.json"

try:
    result = read_mapdb_h5(h5_path)
    save_results(result)
    print("Function executed successfully.")
except Exception as e:
    error_message = str(e)
    print(f"An error occurred: {error_message}")




