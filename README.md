
## 📚 Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Container 1: Metadata Splitting and Cleaning](#container-1)
  - [🚀 Run Container 1](#run-container-1)
    - [1. Split the Metadata File](#split-the-metadata-file)
    - [2. Fetch Ontologies](#fetch-ontologies)
    - [3. Clean Metadata Files](#clean-metadata-files)
    - [4. Check Metadata Size Reduction](#check-metadata-size-reduction)
    - [5. Analyze Metadata Fields Distribution](#analyze-metadata-fields-distribution)
    - [6. Parse Latitude and Longitude](#parse-latitude-and-longitude)
- [Container 2: Benchmark Annotation via Interactive Interface](#container-2)
  - [🚀 Run Container 2](#run-container-2)
- [Container 3: GPT Interaction](#container-3)
  - [🚀 Run Container 3](#run-container-3)
    - [1. Synchronous GPT Interaction](#synchronous-gpt-interaction)
    - [2. Asynchronous GPT Interaction](#asynchronous-gpt-interaction)
    - [3. Create Embeddings](#create-embeddings)
- [Container 4: GPT performance evaluation](#container-4)
  - [🚀 Run Container 4](#run-container-4)
    - [1. GPT runs evaluation](#gpt-runs-evaluation)
    - [2. Overall gpt performance](#overall-gpt-performance)
    - [3. Convert coordinates to places](#convert-coordinates-to-places)
    - [4. Geographic location: GPT versus metadata](#geographic-location-gpt-versus-metadata)




---
<a name="introduction"></a>
# Introduction: 

This repository contains a modular, containerized pipeline for processing and annotating environmental sample metadata. Each container encapsulates a specific set of tasks — as you see depicted below - from preprocessing metadata (C1 - 🔴 red), creating a benchmark dataset (C2 - 🟠 orange), interacting with GPT models (C3 - 🟣 purple), to GPT output evaluation (C3 - 🟢 green). 


![Pipeline overview](pipeline.png)


Let's get what we need to start! 

---
<a name="requirements"></a>
## 📦 Requirements: 


### 1) Clone the repo into your home directory:

```
mkdir ~/github
cd ~/github
git clone https://github.com/GaioTransposon/metadata_mining.git
```

### 2) Download large files and move folder: 

- Download the directory containing all the files you need from [here](https://zenodo.org/records/16100607) to your Downloads folder
- Move to your Downloads folder `cd ~/Downloads`
- Decompress the directory `unzip MicrobeAtlasProject_Zenodo.zip`
- Rename it "MicrobeAtlasProject" `mv MicrobeAtlasProject_Zenodo MicrobeAtlasProject`
- Move it to your home directory `mv MicrobeAtlasProject ~/.`


### 3) Hence, ensure the following directories exist on your machine: 

```
for dir in ~/MicrobeAtlasProject ~/github/metadata_mining/scripts; do
    [ -d "$dir" ] && echo "✅ Exists: $dir" || echo "❌ Missing: $dir"
done
```

### 4) Install Docker

Download and install Docker Desktop: 

- [Download Docker Desktop](https://www.docker.com/products/docker-desktop) (macOS/Windows)
- [Install Docker Engine](https://docs.docker.com/engine/install/) (Linux)

### 5) Verify the installation and launch Docker: 

```
docker --version
open -a Docker
```

### 6) Build the docker image: 

```
cd ~/github/metadata_mining
docker build -t metadmin .
```

### 6.1) Alternatively, you can directly pull the docker image: 

Prebuilt images are available on Docker Hub:

- [gaiotransposon/metadmin](https://hub.docker.com/r/gaiotransposon/metadmin)

Pull with:

```
docker pull gaiotransposon/metadmin:latest
```


### 7) Get your own API keys: 

In order to run container 3, you will need to acquire your own OpenAI API key. Instructions on how to make one you will find [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key). Once you have your API key, place it in `~/MicrobeAtlasProject`. 

Ideally, you generate two separate keys: one for the chat completion (annotation of metadata), and one for creating embeddings. In this pipeline the two are named: `my_api_key` and `my_api_key_embeddings`. The reason for using two separate keys is keeping track of usage quotas for each task. 

In order to run (the last script of) container 4, you will need to acquire your own Google Maps API key (follow instructions [here](https://developers.google.com/maps/documentation/embed/get-api-key?setupProd=configure)). Google provides a free usage tier, which can cover a significant number of API requests. Generate one and name it `google_maps_api_key`. Place it in `~/MicrobeAtlas`. 


<a name="container-1"></a>
## Container 1: Metadata Splitting and Cleaning

The first container provides an environment to run all scripts related to processing and cleaning metagenomic environmental metadata, including coordinates parsing, ontology code translation, and perform some exploratory analysis of the metadata.

---

<a name="run-container-1"></a>
### 🚀 Run Container 1: 

<a name="split-the-metadata-file"></a>
#### 1. Split the large metadata file 🧾 into individual files: 

```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/dirs.py \
    --input_file sample.info.gz \
    --output_dir sample_info_split_dirs \
    --figure_path files_distribution_in_dirs.pdf
```
⚠️ Docker might get fussy with resource limits. In that case you can either 1. increase Docker memory, or 2. (easier) run the same script outside of Docker. 

<a name="fetch-ontologies"></a>
#### 2. Fetch ontologies  🌐: 

```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/fetch_and_join_ontologies.py \
    --wanted_ontologies FOODON ENVO UBERON PO \
    --output_file ontologies_dict
```

<a name="clean-metadata-files"></a>
#### 3. Clean metadata files and replace ontology codes with labels  🧼: 

Increase the file descriptor limit first. By default, many operating systems limit how many files can be open at once. Since this script processes many files in parallel, you must increase the ulimit:

```
ulimit -n 200000
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/clean_and_envo_translate.py \
    --ontology_dict ontologies_dict.pkl \
    --metadata_dirs sample_info_split_dirs \
    --max_processes 8
```

<a name="check-metadata-size-reduction"></a>
#### 4. Check metadata size reduction 📉 : 

This script compares file sizes before and after cleaning and estimates the token-level reduction after the cleaning. It calculates token reduction using bootstrap sampling (default: 100 iterations × 100 samples).

```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/check_metadata_sizes.py
```

<a name="analyze-metadata-fields-distribution"></a>
#### 5. Analyze metadata fields distribution 🧠 : 

This script examines in which metadata fields the benchmark sub-biome information appears. It scans the cleaned metadata files and checks whether the sub-biome (e.g. human gut, sediment, leaf) is found fully or partially in each metadata field. This helps identify the most informative fields across samples and biomes. It outputs a plot and csv summaries with the top-matching fields, based on 1000 random files. 

```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/field_distrib_analysis.py \
    --gold_dict gold_dict.pkl 
```

<a name="parse-latitude-and-longitude"></a>
#### 6. Parse latitude and longitude 🌍: 

```
docker run --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/parse_lat_lon_from_metadata.py \
    --reversal_file samples_with_lat_lon_reversal.tsv \
    --metadata_file metadata.out \
  | grep '^OUTPUT:' \
  | cut -f1-5 \
  | tr '\t' ' ' \
  | sed 's/  */ /g' \
  | sed 's/ *$//' \
  > ~/MicrobeAtlasProject/sample.coordinates.reparsed.filtered
```

<a name="container-2"></a>
## Container 2: Benchmark Annotation via Interactive Interface

This container supports the creation and manual curation of a benchmark (also referred to as gold standard dictionary - file: gold_dict.pkl), which maps selected sample IDs to: 
- a biome (animal, plant, soil, water, other)
- a specific sample origin (sub-biome)
- geographic coordinates (latitude/longitude)
- a short geographic location description

This container includes two interactive scripts: 
- make_gold_dict.py: you are shown metadata from each sample and you can annotate samples yourself.
- edit_gold_dict.py: you can modify or correct existing entries (when you realise you made a mistake).

⚠️ These scripts use input() prompts, so they must be run inside an interactive Docker session as running them directly with conda run or piping won't work properly.


---

<a name="run-container-2"></a>
### 🚀 Run Container 2: 

Launch Docker container interactively:

```
docker run -it --rm \
  --entrypoint bash \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin
```

Activate the environment inside the container:

```
conda activate metadmin_env
```

To create the benchmark from scratch or to continue building on it, run: 

```
python /app/scripts/make_gold_dict.py
```

This starts a session where you can annotate samples one by one. Your progress is automatically saved to gold_dict.pkl.

To edit entries in the existing dictionary, run:

```
python /app/scripts/edit_gold_dict.py
```

To exit either session just type:
`exit`

💾 In both cases your changes are automatically saved to `~/MicrobeAtlasProject/gold_dict.pkl`. 



---

<a name="container-3"></a>
## Container 3: GPT interaction

This container handles all steps related to GPT-based annotation of metadata, including: synchronous or asynchronous interactions with the OpenAI API, preparing and submitting batch jobs (asynchronous runs), fetching responses (asynchronous runs), generating sub-biome embeddings from GPT outputs and from benchmark data. 

⚠️ Before running this container, you will need to acquire your API key and place it in `~/MicrobeAtlasProject`. You could generate two separate keys: one for the chat completion (annotation of metadata), one for creating embeddings. In this pipeline the two are named: `my_api_key` and `my_api_key_embeddings`. The reason for using two separate keys is keeping track of usage quotas for each task. 


<a name="run-container-3"></a>
### 🚀 Run Container 3: 

You can:

- run synchronous interaction with OpenAI (single step)

AND/OR

- run an asynchronous interaction via the batch API (two steps)

Then, you can generate embeddings from GPT results and from benchmark data.

<a name="synchronous-gpt-interaction"></a>
#### Synchronous GPT interaction: 

This script performs end-to-end metadata annotation in a single script using synchronous OpenAI API requests.


```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/openai_main.py \
    --work_dir . \
    --input_gold_dict gold_dict.pkl \
    --n_samples_per_biome 5 \
    --chunking no \
    --chunk_size 2000 \
    --seed 22 \
    --directory_with_split_metadata sample_info_split_dirs \
    --system_prompt_file openai_system_better_prompt_json.txt \
    --encoding_name cl100k_base \
    --api_key_path my_api_key \
    --model gpt-3.5-turbo-1106 \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --max_requests_per_minute 3500 \
    --opt_text normal \
    --output_format json
```

<a name="asynchronous-gpt-interaction"></a>
#### Asynchronous GPT interaction (2 steps): 

Use this if you want to take advantage of OpenAI's batch API for more efficient, large-scale requests.

- Step 1: Submit Batch Job (async requests)

This script prepares metadata and submits it as an OpenAI batch job.



```
  docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/gpt_async_batch.py \
    --work_dir . \
    --input_gold_dict gold_dict.pkl \
    --n_samples_per_biome 5 \
    --chunking "no" \
    --chunk_size 3000 \
    --seed 22 \
    --directory_with_split_metadata "sample_info_split_dirs" \
    --system_prompt_file "openai_system_prompt.txt" \
    --encoding_name "cl100k_base" \
    --api_key_path "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --output_format "inline"
```

- Step 2: Fetch Batch Results

After your batch job is submitted, OpenAI typically processes it within a few minutes to a few hours. However, the maximum processing time is 24 hours. If your job hasn't completed within that window, it will expire, and you'll need to resubmit it.

We have successfully submitted up to 700,000 metadata samples per day and consistently received results well within 24 hours.

To fetch and save completed results locally, run:


```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/gpt_async_fetch_and_save.py \
    --work_dir . \
    --api_key_path my_api_key

```

<a name="create-embeddings"></a>
#### Create Embeddings: 

This script creates text-embedding-3-small vectors from:

- GPT-generated sub-biomes (gpt_clean_output*.csv / .txt)
- Your benchmark sub-biomes (gold_dict.pkl)


```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/embeddings_from_sb.py \
    --directory_path . \
    --api_key_path my_api_key_embeddings \
    --gold_dict_path gold_dict.pkl \
    --embed_model text-embedding-3-small
```


---

<a name="container-4"></a>
## Container 4: GPT performance evaluation

This container evaluates GPT performance by comparing biomes and sub-biomes annotations by GPT (.tsv files for biomes and .json files for sub-biomes) against those of the benchmark (gold_dict.pkl). It does so for each GPT run. For biomes annotation evaluation, it compares strings for either a lenient or an exact match. It produces a summary CSV with per-run biome agreement metrics. For sub-biomes annotation evaluation, it uses embeddings of GPT runs versus embeddings of the benchmark. Then, it computes cosine similarity between sample-ID-matched embeddings, it calculates the distribution of similarities versus the background, and it produces a summary CSV with per-run sub-biome similarity metrics. Pairwise statistical comparisons are performed. Additionally, in this container, we evaluate geographic annotations by GPT by comparing them to the metadata-extracted coordinates. 


<a name="run-container-4"></a>
### 🚀 Run Container 4: 

Four scripts to run:

- GPT runs evaluation: validate_biomes_subbiomes.py
- Overall GPT performance: overall_analysis.py
- Convert coordinates to places: coord_to_text.py
- Geographic location - GPT versus metadata: geo_check.py

⚠️ In order to run the last script of this container you will need a free Google Maps API key (follow instructions [here](https://developers.google.com/maps/documentation/embed/get-api-key?setupProd=configure). Generate one and name it `google_maps_api_key`. Place it in `~/MicrobeAtlas`. 

<a name="gpt-runs-evaluation"></a>
#### GPT runs evaluation: 

This script compares biome and sub-biome annotations per GPT run against the benchmark's. In this manner the performance of GPT runs in which different settings were used (creativity parameters or other parameters), can be compared. You can use my --map_file (gpt_file_label_map.tsv) if you are reproducing my results (LLM performance). If you have your own LLM files you will need to edit gpt_file_label_map.tsv accordingly to reflect your file names and your labels of choice. 


```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python /app/scripts/validate_biomes_subbiomes.py \
    --map_tsv gpt_file_label_map.tsv \
    --gold_dict gold_dict.pkl \
    --embedding_models text-embedding-3-small,Qwen-Qwen3-Embedding-8B,Qwen-Qwen3-Embedding-4B,Qwen-Qwen3-Embedding-0.6B
```

One result and one stats file will be produced for each embedding model (biome_subbiome_results_{embedding_model_name}.csv and biome_subbiome_stats_{embedding_model_name}.csv, respectively)


<a name="overall-gpt-performance"></a>
#### Overall GPT performance: 

This script assesses the overall performance of all GPT runs against the benchmark. You can choose not to include certain files to the overall analysis by adding them to the overall_analysis_excluded_files.txt file. 

```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  -e WORK_DIR=/MicrobeAtlasProject \
  -e SCRIPTS_DIR=/app/scripts \
  metadmin \
    python /app/scripts/overall_analysis.py \
              --work_dir . \
              --metadata_dir sample_info_split_dirs \
              --keyword_based_annot_file keywordsbased_biomes_parsed.csv \
              --exclude_files overall_analysis_excluded_files.txt
```

Results (and stats) will print out to the console. 

<a name="convert-coordinates-to-places"></a>
#### Convert coordinates to places: 

This script performs reverse geocoding on a set of unique latitude/longitude coordinates. This means it converts each coordinate pair into a human-readable place name (e.g.: a city, region, or country). It uses the Nomatin geocoding service OpensStreetMap. It may take long to run as we are using the free version (no API). Approximately you can expect it to take 1.3 seconds per coordinates pair. 


```
docker run -it --rm \
  -e PYTHONUNBUFFERED=1 \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  python -u /app/scripts/coord_to_text.py \
    --work_dir . \
    --coordinates_file sample.coordinates.reparsed.filtered \
    --output_file geocoded_coordinates.csv \
    --min_delay_seconds 1.3
```

You can check the progress by running from another terminal: 

```
tail -f ~/MicrobeAtlasProject/geocoding_progress.log
```


<a name="geographic-location-gpt-versus-metadata"></a>
#### Geographic location: GPT versus metadata: 

This script needs to run interactively because it gives you the possibility to evaluate a set of GPT geographic locations versus the extracted coordinates. You will pick "who" was correct: coordinates-derived geographic location (from metadata) or GPT-derived geographic location. This will help you qualify the mismatches between the two. Start by launching the Docker container interactively:

```
docker run -it --rm \
  --entrypoint bash \
  -v ~/MicrobeAtlasProject:/MicrobeAtlasProject \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin
```

Activate the environment inside the container, then run the script:

```
conda activate metadmin_env
python /app/scripts/geo_check.py \
    --work_dir . \
    --metadata_dir MicrobeAtlasProject/sample_info_split_dirs \
    --api_key_file google_maps_api_key \
    --coordinates_file sample.coordinates.reparsed.filtered \
    --translated_coordinates geocoded_coordinates.csv \
    --random_misclassified_samples_dict random_misclassified_samples_dict.pkl \
    --output_map_all_matches map_with_color_coded_points_all.html \
    --output_map_all_mismatches map_with_color_coded_points_mismatches.html
```

⚠️ If you do not want to play the game when prompted to, you don't have to. Just type `QUIT` and the script will use the already evaluated answers from random_misclassified_samples_dict.pkl. 

To exit the session just type `exit`







