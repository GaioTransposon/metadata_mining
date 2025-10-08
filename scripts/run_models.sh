#!/usr/bin/env bash
set -euo pipefail


# ---------------------------
# Paths
# ---------------------------
MAP_HOST="/mnt/mnemo9/mpelus/dany_paper/MicrobeAtlasProject_Zenodo"
REPO="/mnt/mnemo9/mpelus/dany_paper/metadata_mining"
cd "$MAP_HOST"


# ---------------------------
# Common parameters
# ---------------------------
NSPB=200
CHUNKING="no"
CHUNK_SIZE=2000
SEED=22
TEMP=1.0
MAX_TOKENS=4096
TOP_P=0.75
FREQ_P=0.25
PRES_P=1.5
RPM=120
OPT_TEXT="normal"
SYSTEM_PROMPT="openai_system_prompt_json.txt"
ENCODING="cl100k_base"

# Keys (relative to MAP_HOST)
OPENAI_KEY="api_key_openai.txt"   # OpenAI models
OSS_KEY="api_key_deepinfra.txt"             # OpenAI-compatible OSS endpoint models

# Provider base URLs
OPENAI_BASE_URL=""                                    # empty => default OpenAI
OSS_BASE_URL="https://api.deepinfra.com/v1/openai"    # OpenAI-compatible

# Embedding models
OPENAI_EMBED_MODEL="text-embedding-3-large"
OSS_EMBED_MODEL="Qwen/Qwen3-Embedding-8B"

# ---------------------------
# Model sets
# ---------------------------

# "gpt-3.5-turbo-0125"
# "gpt-3.5-turbo-1106"
# "gpt-5-2025-08-07" doesnt work becuase max_tokens is no longer a parameter, it needs max_token_completions
OPENAI_MODELS=(
  
  
)

OSS_MODELS=(
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "microsoft/phi-4"
)

# ---------------------------
# Pre-flight checks
# ---------------------------
[[ -f "$OPENAI_KEY" ]] || { echo "❌ Missing $OPENAI_KEY in $MAP_HOST"; exit 1; }
[[ -f "$OSS_KEY"    ]] || { echo "❌ Missing $OSS_KEY in $MAP_HOST"; exit 1; }
[[ -d "sample_info_split_dirs" ]] || { echo "❌ Missing sample_info_split_dirs"; exit 1; }
[[ -f "gold_dict.pkl" ]] || { echo "❌ Missing gold_dict.pkl"; exit 1; }
[[ -f "$REPO/scripts/openai_main.py" ]] || { echo "❌ Missing openai_main.py at $REPO/scripts"; exit 1; }
[[ -f "$REPO/scripts/embeddings_from_sb.py" ]] || { echo "❌ Missing embeddings_from_sb.py at $REPO/scripts"; exit 1; }

log(){ printf "\n[%s] %s\n" "$(date +'%F %T')" "$*"; }

run_one () {
  local provider="$1"  # openai|oss
  local model="$2"
  local key_file base_url embed_model
  if [[ "$provider" == "openai" ]]; then
    key_file="$OPENAI_KEY"; base_url="$OPENAI_BASE_URL"; embed_model="$OPENAI_EMBED_MODEL"
  else
    key_file="$OSS_KEY";    base_url="$OSS_BASE_URL";    embed_model="$OSS_EMBED_MODEL"
  fi

  log "=== C3 run :: provider=$provider :: model=$model ==="
  python "$REPO/scripts/openai_main.py" \
    --work_dir . \
    --input_gold_dict gold_dict.pkl \
    --n_samples_per_biome "$NSPB" \
    --chunking "$CHUNKING" \
    --chunk_size "$CHUNK_SIZE" \
    --seed "$SEED" \
    --directory_with_split_metadata sample_info_split_dirs \
    --system_prompt_file "$SYSTEM_PROMPT" \
    --encoding_name "$ENCODING" \
    --api_key_path "$key_file" \
    --model "$model" \
    ${base_url:+--base_url "$base_url"} \
    --temperature "$TEMP" \
    --max_tokens "$MAX_TOKENS" \
    --top_p "$TOP_P" \
    --frequency_penalty "$FREQ_P" \
    --presence_penalty "$PRES_P" \
    --max_requests_per_minute "$RPM" \
    --opt_text "$OPT_TEXT" \
    --output_format json

  log "C3 done for $model → starting embeddings"
  python "$REPO/scripts/embeddings_from_sb.py" \
    --directory_path . \
    --api_key_path "$key_file" \
    --gold_dict_path gold_dict.pkl \
    ${base_url:+--base_url "$base_url"} \
    --embed_model "$embed_model"

  log "Embeddings done for $model"
}

# ---------------------------
# Run sweeps
# ---------------------------
for m in "${OPENAI_MODELS[@]}"; do
  run_one "openai" "$m"
done

for m in "${OSS_MODELS[@]}"; do
  run_one "oss" "$m"
done

log "✅ All runs finished."
