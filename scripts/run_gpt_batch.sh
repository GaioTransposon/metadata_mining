#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


echo -e "\033[32m 35. metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 1 \
    --chunking "no" \
    --chunk_size 2000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base"

sleep 1m
echo -e "\033[32m 36. async"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 





