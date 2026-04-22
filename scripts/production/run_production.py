#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:51:28 2026

@author: dgaio
"""

# copy and zip David's file:
gzip -c /mnt/mnemo6/dpatsch/data_pipeline/downloads/20240219/samples.info \
> /mnt/mnemo5/dgaio/MicrobeAtlasProject/sample.info.gz
# uncompressed file has got 1160616550 lines. 
# NB: file should have a blank line in between a sample s metadata and the next (I edited sample.info.gz to conform to this)

# copy Janko's file:
cp /mnt/mnemo6/janko/projects/microbe_atlas/results/all_minfilt_sampids_map2024.tsv /mnt/mnemo5/dgaio/MicrobeAtlasProject/.

# copy both to local 

# make a test file: 
gzip -dc sample.info.gz | head -n 1000000 > sample.info_test
gzip sample.info_test





python ~/github/metadata_mining/scripts/dirs.py \
    --input_file '~/MicrobeAtlasProject2026/sample.info_test.gz' \
    --output_dir '~/MicrobeAtlasProject2026/sample_info_split_dirs' \
    --figure_path '~/MicrobeAtlasProject2026/files_distribution_in_dirs.pdf'
    
    
python ~/github/metadata_mining/scripts/fetch_and_join_ontologies.py \
    --wanted_ontologies FOODON ENVO UBERON PO \
    --output_file "ontologies_dict" \
    --output_dir ~/MicrobeAtlasProject2026


# $ ulimit -n 200000 <-- it's an estimation derived from: 
# 40 (dirs and cpus at a time) * 3800 (files per dir) = 152000 --> round up: 200000



# CONTINUE FROM HERE: clean script is not translating I think...
    
    
    
python ~/github/metadata_mining/scripts/clean_and_envo_translate.py \
    --path_to_dir "~/MicrobeAtlasProject2026" \
    --ontology_dict "ontologies_dict.pkl" \
    --metadata_dirs "sample_info_split_dirs" \
    --max_processes 8
# or on atlas: --max_processes 40







python ~/github/metadata_mining/scripts/production/generate_sample_list.py \
    --work_dir "cloudstor/Gaio/MicrobeAtlasProject" \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --seed 22 \
    --output_file "samples_list_202604.txt" \
    --whitelist_file "all_minfilt_sampids_map2021.tsv"
    
    
python ~/github/metadata_mining/scripts/production/gpt_async_batch_production.py \
    --work_dir "cloudstor/Gaio/MicrobeAtlasProject" \
    --sample_list_file "missing_samples.txt" \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --output_pkl "metadataprov.pkl" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_better_prompt_batch.txt" \
    --api_key_path "Desktop/keys/my_api_key_production_run" \
    --model "gpt-3.5-turbo-0125" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --n_samples 9000 \
    --n_batches 5 \
    --delay_minutes 1.5 \
    --state_file "state_file_202504.txt"
    



