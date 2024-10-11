#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


# =============================================================================
# #######
# 
# # Necessary before running openai in the sync mode
# 
# yes | pip uninstall openai 
# 
# yes | pip install openai==0.28
# 
# 
# #######
# 
# 
# 
# 
# echo -e "\033[32m 1. sync chunking yes/no"
# echo -e "\033[0m"
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 2. sync chunking yes/no"
# echo -e "\033[0m"
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 3. chunking sizes"
# echo -e "\033[0m"
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 1000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 4. chunking sizes"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 2000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 5. chunking sizes"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 4000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
#     
#     
#     
# sleep 1m
# echo -e "\033[32m 6. chunking sizes"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 5000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
#     
#     
#     
# sleep 1m
# echo -e "\033[32m 7. chunking sizes"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 6000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
#     
#     
#     
# sleep 1m
# echo -e "\033[32m 8. model"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep"
#     
# 
# 
# sleep 1m
# echo -e "\033[32m 9. model"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-0125" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 10. model"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-0125" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep"
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 11. model"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-4-0613" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 12. model"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-4-0613" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep"
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 13. temp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 0.5 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 14. temp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.5 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 15. temp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 2.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 16. topp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.0 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 17. topp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.5 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 18. topp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 1.0 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 19. freqp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.0 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 20. freqp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 1.0 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 21. freqp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 2.0 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 22. presp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 0.0 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 23. presp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.0 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 24. presp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 2.0 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 25. temp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 0.5 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 26. temp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.5 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 27. temp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 2.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 28. topp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.0 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 29. topp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.5 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 30. topp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 1.0 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 31. freqp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.0 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 32. freqp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 1.0 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 33. freqp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 2.0 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 34. presp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 0.0 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 35. presp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.0 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# sleep 1m
# echo -e "\033[32m 36. presp"
# echo -e "\033[0m"
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.0 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 2.0 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 37. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
#     
#     
# 
# sleep 1m
# echo -e "\033[32m 38. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep1"
#     
#     
#     
# sleep 1m
# echo -e "\033[32m 39. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep2"
#     
#     
#     
# sleep 1m
# echo -e "\033[32m 40. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep3"
#     
#     
#     
# sleep 1m
# echo -e "\033[32m 41. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep4"
#     
#     
#     
# sleep 1m
# echo -e "\033[32m 42. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep5"
#     
#     
#     
# 
# sleep 1m
# echo -e "\033[32m 43. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 32 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 44. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 32 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep"
#     
#     
#     
#     
# sleep 1m
# echo -e "\033[32m 45. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 42 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "normal"
# 
# 
# 
# sleep 1m
# echo -e "\033[32m 46. robustness"
# echo -e "\033[0m"
# 
# 
# 
# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 50 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 42 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text "rep"
# =============================================================================



#######

# Necessary before running openai in the async mode
yes | pip uninstall openai 

yes | pip install openai


#######


echo -e "\033[32m metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base"


sleep 1m
echo -e "\033[32m 47. async temp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 



sleep 1m
echo -e "\033[32m 48. async temp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 0.5 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 49. async temp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.5 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 



sleep 1m
echo -e "\033[32m 50. async temp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 2.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 51. async topp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 




sleep 1m
echo -e "\033[32m 52. async topp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.0 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    
sleep 1m
echo -e "\033[32m 53. async topp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.5 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    
sleep 1m
echo -e "\033[32m 54. async topp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 1.0 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    

sleep 1m
echo -e "\033[32m 55. async freqp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    


sleep 1m
echo -e "\033[32m 56. async freqp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.0 \
    --presence_penalty 1.5 



sleep 1m
echo -e "\033[32m 57. async freqp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 1.0 \
    --presence_penalty 1.5 
    
    

sleep 1m
echo -e "\033[32m 58. async freqp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 2.0 \
    --presence_penalty 1.5 
    
    


sleep 1m
echo -e "\033[32m 59. async presp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    
sleep 1m
echo -e "\033[32m 60. async presp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 0.0
    
    

sleep 1m
echo -e "\033[32m 61. async presp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.0
    
    
sleep 1m
echo -e "\033[32m 62. async presp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 2.0
    


#############################################################################



    
echo -e "\033[32m metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base"




sleep 1m
echo -e "\033[32m 63. robustness"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    
  sleep 1m
  echo -e "\033[32m 64. robustness"
  echo -e "\033[0m"

  python github/metadata_mining/scripts/gpt_async_batch.py \
      --work_dir "MicrobeAtlasProject" \
      --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
      --api_key_path "my_api_key" \
      --model "gpt-3.5-turbo-1106" \
      --temperature 1.00 \
      --max_tokens 4096 \
      --top_p 0.75 \
      --frequency_penalty 0.25 \
      --presence_penalty 1.5 
      
      

sleep 1m
echo -e "\033[32m 65. robustness"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    
sleep 1m
echo -e "\033[32m 66. robustness"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    
    
sleep 1m
echo -e "\033[32m 67. robustness"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    
sleep 1m
echo -e "\033[32m 68. robustness"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    

echo -e "\033[32m metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 32 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base"

sleep 1m
echo -e "\033[32m 69. robustness"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    

sleep 1m
echo -e "\033[32m 70. robustness"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 



echo -e "\033[32m metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 42 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base"

sleep 1m
echo -e "\033[32m 71. robustness"
echo -e "\033[0m"



python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    

sleep 1m
echo -e "\033[32m 72. robustness"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    
    
    
echo -e "\033[32m metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 50 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_please.txt" \
    --encoding_name "cl100k_base"



sleep 1m
echo -e "\033[32m 73. please"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_please.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    


sleep 1m
echo -e "\033[32m 74. please"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_please.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 