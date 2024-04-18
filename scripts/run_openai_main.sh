#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


echo -e "\033[32mStart 1st test - chunking yes/no"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
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


# sleep 1m
# echo -e "\033[32mStart 2nd test - chunking yes/no"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
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
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 3rd test - chunking sizes"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
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
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 4th test - chunking sizes"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
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
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 5th test - different model"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "yes" \
#     --chunk_size 2000 \
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
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
    
# sleep 1m
# echo -e "\033[32mStart 6th test - temperatures"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
#     --chunk_size 2000 \
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
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 7th test - temperatures"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
#     --chunk_size 2000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.50 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 8th test - temperatures"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
#     --chunk_size 2000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 2.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 9th test - topp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
#     --chunk_size 2000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.00 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 10th test - topp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
#     --chunk_size 2000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.50 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 11th test - topp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
#     --chunk_size 2000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 1.00 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 12th test - freqp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
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
#     --frequency_penalty 0.00 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 13th test - freqp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
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
#     --frequency_penalty 1.00 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"

# sleep 1m
# echo -e "\033[32mStart 14th test - freqp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
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
#     --frequency_penalty 2.00 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 15th test - presp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
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
#     --presence_penalty 0.0 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 16th test - presp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
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
#     --presence_penalty 1.0 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 17th test - presp"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunking "no" \
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
#     --presence_penalty 2.0 \
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 18th test - random seed"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 2000 \
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
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    
# sleep 1m
# echo -e "\033[32mStart 19th test - random seed"
# echo -e "\033[0m"

# python ~/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunking "yes" \
#     --chunk_size 2000 \
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
#     --max_requests_per_minute 10000 \
#     --opt_text "normal"
    