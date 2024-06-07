#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


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