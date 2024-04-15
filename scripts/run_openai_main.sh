#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 200 \
    --chunking "yes" \
    --chunk_size 2000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"