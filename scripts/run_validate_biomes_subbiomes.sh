#!/bin/bash

SCRIPT_DIR="/Users/dgaio/github/metadata_mining/scripts"
PYTHON_SCRIPT="${SCRIPT_DIR}/validate_biomes_subbiomes.py"
cd $SCRIPT_DIR

function activate_conda {
    CONDA_PATH=$(type -p conda)
    source "$(dirname "$CONDA_PATH")/../etc/profile.d/conda.sh"
    conda activate spyder_env
    echo "Activated conda environment: spyder_env"
}


activate_conda


#####################################################################################################
# 1. sync - chunking y/n
#####################################################################################################

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_normal_dt202407191713.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API117_normal_dt202407191720.txt"
)

LABELS=(
    "sync_chunkN"
    "sync_chunkY"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################
# 2. sync - chunksizes
#####################################################################################################

FILES=(
    "gpt_clean_output_nspb100_chunkingyes_chunksize6000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API47_normal_dt202410151045.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize5000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API51_normal_dt202407191744.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize4000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API61_normal_dt202407191736.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API78_normal_dt202407191729.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API117_normal_dt202407191720.txt"
)


LABELS=(
    "sync_chunkY_size6000"
    "sync_chunkY_size5000"
    "sync_chunkY_size4000"
    "sync_chunkY_size3000"
    "sync_chunkY_size2000"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"



#####################################################################################################
# 3. sync - models
#####################################################################################################

FILES=(
    "gpt_clean_output_nspb50_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API60_normal_dt202407241527.txt"
    "gpt_clean_output_nspb50_chunkingyes_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API57_normal_dt202407241530.txt"
    "gpt_clean_output_nspb50_chunkingyes_chunksize2000_modelgpt-4-0613_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API58_normal_dt202407241543.txt"
    "gpt_clean_output_nspb50_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API58_repeat_dt202407241550.txt"
    "gpt_clean_output_nspb50_chunkingyes_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API58_repeat_dt202407241553.txt"
    "gpt_clean_output_nspb50_chunkingyes_chunksize2000_modelgpt-4-0613_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API57_repeat_dt202407241607.txt"
)


LABELS=(
    "gpt3.5-turbo-1106"
    "gpt3.5-turbo-0125"
    "gpt4-0613"
    "gpt3.5-turbo-1106_rep"
    "gpt3.5-turbo-0125_rep"
    "gpt4-0613_rep"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"



#####################################################################################################
# 4. creativity params testing, synchronous requests, chunking: 
#####################################################################################################

# temp

FILES=(
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp0.5_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API117_normal_dt202406051402.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.5_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051409.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp2.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051415.txt"
)


LABELS=(
    "sync_chunkY_temp1.0"
    "sync_chunkY_temp0.5"
    "sync_chunkY_temp1.5"
    "sync_chunkY_temp2.0"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################

# topp

FILES=(
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.0_freqp0.25_presp1.5_rs22_API118_normal_dt202406051422.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.5_freqp0.25_presp1.5_rs22_API118_normal_dt202406051428.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp1.0_freqp0.25_presp1.5_rs22_API120_normal_dt202406051435.txt"
)


LABELS=(
    "sync_chunkY_topp0.75"
    "sync_chunkY_topp0.0"
    "sync_chunkY_topp0.5"
    "sync_chunkY_topp1.0"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################


# freqp

FILES=(
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp1.0_presp1.5_rs22_API132_normal_dt202406051450.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt"
)


LABELS=(
    "sync_chunkY_freqp0.25"
    "sync_chunkY_freqp0.0"
    "sync_chunkY_freqp1.0"
    "sync_chunkY_freqp2.0"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################

# presp

FILES=(
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp0.0_rs22_API118_normal_dt202406051507.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.0_rs22_API118_normal_dt202406051513.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp2.0_rs22_API118_normal_dt202406051520.txt"
)


LABELS=(
    "sync_chunkY_presp1.5"
    "sync_chunkY_presp0.0"
    "sync_chunkY_presp1.0"
    "sync_chunkY_presp2.5"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################
# 5. creativity params testing, synchronous requests, not chunking: 
#####################################################################################################

# temp

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_normal_dt202406051335.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp2.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_repeat_dt202406131512.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.5_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_repeat_dt202406131503.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp0.5_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_repeat_dt202406131455.txt"
)


LABELS=(
    "sync_chunkN_temp1.0"
    "sync_chunkN_temp2.0"
    "sync_chunkN_temp1.5"
    "sync_chunkN_temp0.5"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################

# topp

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_normal_dt202406051335.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp1.0_freqp0.25_presp1.5_rs22_API499_repeat_dt202406131538.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.5_freqp0.25_presp1.5_rs22_API499_repeat_dt202406131529.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.0_freqp0.25_presp1.5_rs22_API499_repeat_dt202406131521.txt"
)



LABELS=(
    "sync_chunkN_topp0.75"
    "sync_chunkN_topp1.0"
    "sync_chunkN_topp0.5"
    "sync_chunkN_topp0.0"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################


# freqp

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_normal_dt202406051335.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API499_repeat_dt202406171520.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp1.0_presp1.5_rs22_API499_repeat_dt202406171543.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API499_repeat_dt202406171555.txt"
)



LABELS=(
    "sync_chunkN_freqp0.25"
    "sync_chunkN_freqp0.0"
    "sync_chunkN_freqp1.0"
    "sync_chunkN_freqp2.0"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################

# presp

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_normal_dt202406051335.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp0.0_rs22_API499_repeat_dt202406171624.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.0_rs22_API499_repeat_dt202406171636.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp2.0_rs22_API499_repeat_dt202406171647.txt"
)



LABELS=(
    "sync_chunkN_presp1.5"
    "sync_chunkN_presp0.0"
    "sync_chunkN_presp1.0"
    "sync_chunkN_presp2.5"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################
# 6. creativity params testing, asynchronous requests:
#####################################################################################################

# temp

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch531mkNXTyMyYTSBJiWVwcLm7_dt202406071408.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp0.5_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchyTdqYI5NQpzbGR0gNBO6HU6x_dt202406071410.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.5_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch9U6Gzkj2sLyTDWvTJooiQvtI_dt202406071411.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp2.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchyP1BWiaONwk6peX6svVVuimq_dt202406071412.csv"
)



LABELS=(
    "async_temp1.0"
    "async_temp0.5"
    "async_temp1.5"
    "async_temp2.0"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################

# topp

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch531mkNXTyMyYTSBJiWVwcLm7_dt202406071408.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.0_freqp0.25_presp1.5_rs22_batchw7kCe5puq0WqYKQq7BMa5iuT_dt202406071413.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.5_freqp0.25_presp1.5_rs22_batchkumTlIdj4B7wKVARjFoHQe4x_dt202406071414.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp1.0_freqp0.25_presp1.5_rs22_batchudcfHtFjYh6fZMOL0oKk7yLR_dt202406071415.csv"
)



LABELS=(
    "async_topp0.75"
    "async_topp0.0"
    "async_topp0.5"
    "async_topp1.0"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################


# freqp

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch531mkNXTyMyYTSBJiWVwcLm7_dt202406071408.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_batchrTuC2pXWV1UcVrQjzhdQsTVm_dt202406071417.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp1.0_presp1.5_rs22_batch17uVtvnldxiQ6kr5NbBKJInh_dt202406071418.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_batchrsizRxW3st3W2olyp0fU3Go0_dt202406071419.csv"
)



LABELS=(
    "async_freqp0.25"
    "async_freqp0.0"
    "async_freqp1.0"
    "async_freqp2.0"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################

# presp

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch531mkNXTyMyYTSBJiWVwcLm7_dt202406071408.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp0.0_rs22_batchKcsV0wRxs2BFSS1dxWjXoCgq_dt202406071420.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.0_rs22_batchMPSOJiGWXtHSlVNfGJnR28vS_dt202406071421.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp2.0_rs22_batchDs2ASNWkbUmXrtZn2Mlw5dJY_dt202406071422.csv"
)



LABELS=(
    "async_presp1.5"
    "async_presp0.0"
    "async_presp1.0"
    "async_presp2.5"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"



#####################################################################################################
# 7. sync - reproducibility
#####################################################################################################

FILES=(
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API119_repeat_dt202406061246.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API500_timeSat1600CEST_dt202407271555.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API500_timeMon1100CEST_dt202407291058.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API500_timeMon1600CEST_dt202407291559.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_timeTue1100CEST_dt202407301104.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API500_timeTue1100CESTrep_dt202407301116.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_timeTue1600CEST_dt202407301558.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_timeTue1600CESTrep_dt202407301622.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_timeTue1800CEST_dt202407301757.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API503_timeTue1800CESTrep_dt202407301808.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_timeTue2100CEST_dt202407302116.txt"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_timeTue2100CESTrep_dt202407302129.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs32_API118_normal_dt202406051533.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs32_API120_repeat_dt202406061258.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API109_normal_dt202406051527.txt"
    "gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API110_repeat_dt202406061252.txt"
)


LABELS=(
    "sync_inl_rs22"
    "sync_inl_rs22_rep1"
    "sync_inl_rs22_rep2"
    "sync_inl_rs22_rep3"
    "sync_inl_rs22_rep4"
    "sync_inl_rs22_rep5"
    "sync_inl_rs22_rep6"
    "sync_inl_rs22_rep7"
    "sync_inl_rs22_rep8"
    "sync_inl_rs22_rep9"
    "sync_inl_rs22_rep10"
    "sync_inl_rs22_rep11"
    "sync_inl_rs22_rep12"
    "sync_inl_rs32"
    "sync_inl_rs32_rep"
    "sync_inl_rs42"
    "sync_inl_rs42_rep"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"



#####################################################################################################
# 8. async - reproducibility 
#####################################################################################################

FILES=(
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch3UrsPni6Ue63YXRYwisZ5Ygz_dt202407311542.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch4V631Bmz2q8THhl5vWhjN7xL_dt202407311543.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch9Kxb9CgSKO4Jrc4nHeNXjIlL_dt202407311542.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchCIMrqld1ZrfyF7qtmkdMHt7V_dt202407311541.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchWLGIEwamUdCiNXmuJ8ecDIiZ_dt202407311542.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchWXV7DhkQXGAU3ieCwTOvVkrJ_dt202407311542.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchYBPAJfHUxVnBm9YVywZL5wUc_dt202407311543.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchbYb8LE5CosmqIJqySwqu6wyx_dt202407311543.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchkJkjHPS0P9LlbDlUOfaXwotC_dt202407311543.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchkeK2m8Hncyy2FaWDAJwvDyn3_dt202407311543.csv"
    "gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-0125_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batchkxeRGfXerVFAPjV6LvHDMDmu_dt202407311542.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs55_batch3o0hNkYn3qMIb6H3Eqa7ZVpr_dt202407231432.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs55_batchn8C5mQeif6OV1LBhJ5fPd2jt_dt202407231425.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs66_batchZk8tZDOF5WsrqcAvYPewbLtS_dt202407231432.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs66_batchluBKAqRedUv3mOgkmjcJ6Ebq_dt202407231426.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs77_batch8yuPSurBvuXCp8m7AAzpOYMZ_dt202407231427.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs77_batchl8fGIwSNBj0DXJfBHtArnz8n_dt202407231432.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs88_batchNZrsNsi10rMTWPyZUocTHv6r_dt202407231427.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs88_batchVuGTjZgtQSotdEoAWrBVk3cd_dt202407231433.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs99_batchFH19lIBQYqjKQ0shEGRVpqhi_dt202407231427.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs99_batchFZgD75ev26VujM8LYnIt8pFg_dt202407231433.csv"
)


    
LABELS=(
    "async_js_rs22_rep1"
    "async_js_rs22_rep2"
    "async_js_rs22_rep3"
    "async_js_rs22_rep4"
    "async_js_rs22_rep5"
    "async_js_rs22_rep6"
    "async_js_rs22_rep7"
    "async_js_rs22_rep8"
    "async_js_rs22_rep9"
    "async_js_rs22_rep10"
    "async_js_rs22_rep11"
    "async_js_rs55"
    "async_js_rs55_rep"
    "async_js_rs66"
    "async_js_rs66_rep"
    "async_js_rs77"
    "async_js_rs77_rep"
    "async_js_rs88"
    "async_js_rs88_rep"
    "async_js_rs99"
    "async_js_rs99_rep"   
)


    
python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"



#####################################################################################################
# 9. sync vs async all
#####################################################################################################

FILES=(
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e493dbacc8190a39bd001c1f72d6b_dt202410151251.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API250_normal_dt202410151053.txt"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API250_rep1_dt202410151059.txt"
	"gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API250_rep2_dt202410151106.txt"
	"gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API250_rep3_dt202410151112.txt"
	"gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API250_rep4_dt202410151118.txt"
	"gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API250_rep5_dt202410151124.txt"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e49446d888190a86e428fb46406c5_dt202410151251.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e4949656881908d97b5142449b51f_dt202410151251.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e494e783081909e34f78775804f08_dt202410151251.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e49543298819085d1b0b2325ca876_dt202410151252.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e4bc1b1cc8190a3eeff102b680361_dt202410151302.csv"

)


LABELS=(
    "async_js"
    "sync_js"
    "sync_js_rep1"
    "sync_js_rep2"
    "sync_js_rep3"
    "sync_js_rep4"
    "sync_js_rep5"
    "async_js_rep1"
    "async_js_rep2"
    "async_js_rep3"
    "async_js_rep4"
    "async_js_rep5"  
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"


#####################################################################################################
# 10. please
#####################################################################################################

FILES=(
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e493dbacc8190a39bd001c1f72d6b_dt202410151251.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e49446d888190a86e428fb46406c5_dt202410151251.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e4949656881908d97b5142449b51f_dt202410151251.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e494e783081909e34f78775804f08_dt202410151251.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e49543298819085d1b0b2325ca876_dt202410151252.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e4bc1b1cc8190a3eeff102b680361_dt202410151302.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e49abc8ac81908590f00a2e248014_dt202410151253.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_batch670e49b07d388190b3cd6f619f824a3b_dt202410151253.csv"
)


LABELS=(
    "async_js"
    "async_js_rep1"
    "async_js_rep2"
    "async_js_rep3"
    "async_js_rep4"
    "async_js_rep5"  
    "async_js_please"
    "async_js_please_rep"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"



#####################################################################################################
# 11. output formats
#####################################################################################################

FILES=(
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_formatjson_batch67164d8ded048190badb01069142cbda_dt202410211448.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_formatjson_batch67164da1e4ac81909991913126e9eecb_dt202410211448.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_formatinline_batch67164db65fd0819097f31ac6c55e8caf_dt202410211448.csv"
    "gpt_clean_output_nspb50_chunkingno_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_formatinline_batch67164dbbe8b081908bf02a4d2b6e2f65_dt202410211449.csv"
)


LABELS=(
    "json"
    "json_rep"
    "inline"
    "inline_rep"
)

python "$PYTHON_SCRIPT" --files "${FILES[@]}" --labels "${LABELS[@]}"





