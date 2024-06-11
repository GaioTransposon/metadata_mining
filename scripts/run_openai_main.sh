#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


echo -e "\033[32m 1. chunking yes/no"
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
echo -e "\033[32m 2. chunking yes/no"
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
echo -e "\033[32m 3. chunking sizes"
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
echo -e "\033[32m 4. chunking sizes"
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
echo -e "\033[32m 5. model"
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
echo -e "\033[32m 6. temp"
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
echo -e "\033[32m 7. temp"
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
echo -e "\033[32m 8. temp"
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
echo -e "\033[32m 9. topp"
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
echo -e "\033[32m 10. topp"
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
echo -e "\033[32m 11. topp"
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
echo -e "\033[32m 12. freqp"
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
echo -e "\033[32m 13. freqp"
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
echo -e "\033[32m 14. freqp"
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
echo -e "\033[32m 15. presp"
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
echo -e "\033[32m 16. presp"
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
echo -e "\033[32m 17. presp"
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
echo -e "\033[32m 18. rs"
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
echo -e "\033[32m 19. rs"
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



# Day 2: 

echo -e "\033[32m 20. reproducibility 1"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 21. reproducibility 2"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 22. reproducibility 3"
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
    --opt_text "repeat"





sleep 1m
echo -e "\033[32m 23. temp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 24. temp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 25. temp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 26. topp"
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
    --opt_text "repeat"



sleep 1m
echo -e "\033[32m 27. topp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 28. topp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 29. freqp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 30. freqp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 31. freqp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 32. presp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 33. presp"
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
    --opt_text "repeat"


sleep 1m
echo -e "\033[32m 34. presp"
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
    --opt_text "repeat"



# Day 3: 



echo -e "\033[32m 35. metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "no" \
    --chunk_size 2000 \
    --seed 22 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base"

sleep 1m
echo -e "\033[32m 35. async"
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


sleep 1m
echo -e "\033[32m 36. model"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-0125" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 37. temp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 0.5 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 38. temp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.5 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 39. temp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 2.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 40. topp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.0 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 41. topp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.5 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 42. topp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 1.0 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 43. freqp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.00 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 44. freqp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 1.00 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 45. freqp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 2.00 \
    --presence_penalty 1.5 


sleep 1m
echo -e "\033[32m 46. presp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 0.0


sleep 1m
echo -e "\033[32m 47. presp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.0


sleep 1m
echo -e "\033[32m 48. presp"
echo -e "\033[0m"

python github/metadata_mining/scripts/gpt_async_batch.py \
    --work_dir "MicrobeAtlasProject" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.0 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 2.0


sleep 1m
echo -e "\033[32m 49. metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "no" \
    --chunk_size 2000 \
    --seed 42 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base"


sleep 1m
echo -e "\033[32m 49. rs 42"
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

sleep 1m
echo -e "\033[32m 50. metadata prep"
echo -e "\033[0m"

python github/metadata_mining/scripts/metadata_preparation.py \
    --work_dir "MicrobeAtlasProject" \
    --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
    --n_samples_per_biome 100 \
    --chunking "no" \
    --chunk_size 2000 \
    --seed 32 \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
    --encoding_name "cl100k_base"


sleep 1m
echo -e "\033[32m 50. rs 32"
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