#!/usr/bin/env bash



source ~/miniconda3/etc/profile.d/conda.sh

conda activate spyder_env


# End date in format YYYY-MM-DD HH:MM:SS
end_date="2023-07-27 23:00:00"

# Get current date and time
current_date=$(date '+%Y-%m-%d %H:%M:%S')

# Compare current date with the end date
if [[ "$current_date" < "$end_date" ]]; then
    echo "Test started at $(date)" >> test_log.txt
    start_time=$(date +%s)

    # Run your Python script with specified options
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
    --model "gpt-3.5-turbo-0125" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 10000 \
    --opt_text "normal"

    end_time=$(date +%s)
    elapsed_time=$(($end_time - $start_time))
    
    echo "Script running took ${elapsed_time} seconds" >> test_log.txt
    echo "Test ended at $(date)" >> test_log.txt
    echo "--------------------------------------" >> test_log.txt
else
    echo "Past the end date, not running the task." >> test_log.txt
    exit 1
fi