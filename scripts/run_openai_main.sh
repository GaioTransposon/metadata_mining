#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


#######

# Necessary before running openai in the sync mode

yes | pip uninstall openai 

yes | pip install openai==0.28


#######




echo -e "\033[32m 1. sync chunksize 5000"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "yes" \
    --chunk_size 5000 \
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
    --max_requests_per_minute 3500 \
    --opt_text "normal" \
    --output_format 'inline' 




echo -e "\033[32m 2. sync chunksize 6000"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "yes" \
    --chunk_size 6000 \
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
    --max_requests_per_minute 3500 \
    --opt_text "normal" \
    --output_format 'inline' 
    
    


echo -e "\033[32m 3. sync reproducibility"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_json.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 3500 \
    --opt_text "normal" \
    --output_format 'json' 
    
    
    
echo -e "\033[32m 4. sync reproducibility"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_json.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 3500 \
    --opt_text "rep1" \
    --output_format 'json' 
    
    

echo -e "\033[32m 5. sync reproducibility"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_json.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 3500 \
    --opt_text "rep2" \
    --output_format 'json' 
    
    
echo -e "\033[32m 6. sync reproducibility"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_json.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 3500 \
    --opt_text "rep3" \
    --output_format 'json' 
    


echo -e "\033[32m 7. sync reproducibility"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_json.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 3500 \
    --opt_text "rep4" \
    --output_format 'json' 
    
    
    
    
echo -e "\033[32m 8. sync reproducibility"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_json.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 3500 \
    --opt_text "rep6" \
    --output_format 'json' 



# =============================================================================
# #######
# 
# # Necessary before running openai in the async mode
# yes | pip uninstall openai 
# 
# yes | pip install openai
# 
# 
# #######
# 
# 
# echo -e "\033[32m metadata prep"
# echo -e "\033[0m"
# 
# python github/metadata_mining/scripts/metadata_preparation.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base"
# 
# 
# sleep 1m
# echo -e "\033[32m 47. async temp"
# echo -e "\033[0m"
# 
# python github/metadata_mining/scripts/gpt_async_batch.py \
#     --work_dir "MicrobeAtlasProject" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 
# 
# 
# =============================================================================




