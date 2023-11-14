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
from scipy.sparse import save_npz


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


# Function to decode byte strings in a DataFrame
def decode_byte_strings(df):
    return df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Function call with the path to your HDF5 file
result = read_mapdb_h5('/mnt/mnemo3/janko/data/microbe_atlas/hdf5/v0.2.1/metag_minfilter/samples-otus.97.metag.minfilter.minCov90.noMulticell.h5')

# Access the OTU table and metadata
otu_table = result["otu_data"]["otu_table"]
sample_index = result["otu_data"]["sample_index"]
otu_index = result["otu_data"]["otu_index"]
meta_data = result["meta_data"]

# Decode byte strings in sample_index and save to CSV
decoded_sample_index = [s.decode('utf-8') for s in sample_index]
sample_index_df = pd.DataFrame(decoded_sample_index, columns=['sample_index'])
sample_index_df.to_csv('sample_index.csv', index=False)

# Decode byte strings in meta_data and save to CSV
for key, df in meta_data.items():
    decoded_df = decode_byte_strings(df)
    decoded_df.to_csv(f'/mnt/mnemo5/dgaio/metadata_{key}.csv', index=False)



