#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


echo -e "\033[32m chunking yes/no"
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


sleep 1m
echo -e "\033[32m chunking yes/no"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "no" \
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


sleep 1m
echo -e "\033[32m chunking sizes"
echo -e "\033[0m"

python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "yes" \
    --chunk_size 1000 \
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


sleep 1m
echo -e "\033[32m chunking sizes"
echo -e "\033[0m"


python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "yes" \
    --chunk_size 3000 \
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


sleep 1m
echo -e "\033[32m model"
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
    --model "gpt-3.5-turbo-0125" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m temp"
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
    --temperature 0.5 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m temp"
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
    --temperature 1.5 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m temp"
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
    --temperature 2.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m topp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.0 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"



sleep 1m
echo -e "\033[32m topp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.5 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m topp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 1.0 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m freqp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.0 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m freqp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 1.0 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m freqp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 2.0 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m presp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 0.0 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m presp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.0 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m presp"
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
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 2.0 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m rs"
echo -e "\033[0m"


python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "yes" \
    --chunk_size 2000 \
    --seed 42 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"


sleep 1m
echo -e "\033[32m rs"
echo -e "\033[0m"


python ~/github/metadata_mining/scripts/openai_main.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "yes" \
    --chunk_size 2000 \
    --seed 32 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"