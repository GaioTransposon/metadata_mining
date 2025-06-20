# Container 1: Metadata Splitting and Cleaning

This Docker container is part of the **MicrobeAtlasProject** pipeline. It provides a consistent environment to run all scripts related to processing and cleaning environmental metadata, including coordinate parsing, ontology translation, and exploratory analysis.

---

## 📦 Requirements: 


### 1) clone the repo into your home directory:

```
cd ~
git clone link_to_clone_repo
```


### 2) Download large files and move to folder: 

- Make a directory: 
```
cd ~
mkdir MicrobeAtlasProject
```
- Download these large files: ..., ..., ...
- Place them in ~/MicrobeAtlasProject/.


### 3) Ensure the following directories exist on your machine: 

```
~/MicrobeAtlasProject/
~/github/metadata_mining/scripts/
~/github/metadata_mining/source_data/
~/github/metadata_mining/middle_dir/
```

### 4) Install Docker

Download and install Docker Desktop: 

- [Download Docker Desktop](https://www.docker.com/products/docker-desktop) (macOS/Windows)
- [Install Docker Engine](https://docs.docker.com/engine/install/) (Linux)

### 5) Verify the installation with: 

```
docker --version
```

### 6) Launch Docker: 

```
open -a Docker
```

### 7) Build the docker image: 

```
docker build -t metadmin .
```

---


## 🚀 Run the container: 

### 1. Split the metadata file 🧾 into individual files: 

```
conda activate metadmin_env
docker run -it --rm -v ~/MicrobeAtlasProject:/data metadmin \
  conda run -n metadmin_env python /app/scripts/dirs.py \
    --input_file '/data/sample.info_test.gz' \
    --output_dir '/data/sample_info_split_dirs_test' \
    --figure_path '/data/files_distribution_in_dirs_test.pdf'
```

### 2. Fetch ontologies  🌐: 

```
docker run -it --rm -v ~/MicrobeAtlasProject:/data metadmin \
  conda run -n metadmin_env python /app/scripts/fetch_and_join_ontologies.py \
    --wanted_ontologies FOODON ENVO UBERON PO \
    --output_dir '/data' \
    --output_file 'ontologies_dict'
```

### 3. Clean metadata files and replace ontology codes with labels  🧼: 

#### Increase the file descriptor limit first. By default, many operating systems limit how many files can be open at once. Since this script processes many files in parallel, you must increase the ulimit:

```
ulimit -n 200000
docker run -it --rm -v ~/MicrobeAtlasProject:/data metadmin \
  conda run -n metadmin_env python /app/scripts/clean_and_envo_translate.py \
    --path_to_dir "/data" \
    --ontology_dict "ontologies_dict.pkl" \
    --metadata_dirs "sample_info_split_dirs_test" \
    --max_processes 8
```

### 4. Check metadata size reduction 📉 : 

#### This script compares file sizes before and after cleaning and estimates the token-level reduction after the cleaning. It calculates token reduction using bootstrap sampling (default: 100 iterations × 100 samples).

```
docker run -it --rm -v ~/MicrobeAtlasProject:/data metadmin \
  conda run -n metadmin_env python /app/scripts/check_metadata_sizes.py \
    --split_dirs '/data/sample_info_split_dirs_test'
```

### 5. Analyze metadata fields distribution 🧠 : 

#### This script examines in which metadata fields the benchmark sub-biome information appears. It scans the cleaned metadata files and checks whether the sub-biome (e.g. human gut, sediment, leaf) is found fully or partially in each metadata field. This helps identify the most informative fields across samples and biomes. It outputs a plot and csv summaries with the top-matching fields, based on 1000 random files. 

```
docker run -it --rm \
  -v ~/MicrobeAtlasProject:/data \
  -v ~/github/metadata_mining/source_data:/source_data \
  -v ~/github/metadata_mining/scripts:/app/scripts \
  metadmin \
  conda run -n metadmin_env python /app/scripts/field_distrib_analysis.py \
    --split_dirs '/data/sample_info_split_dirs_test/'
```

### 6. Parse latitude and longitude 🌍: 

```
docker run --rm \
  -v ~/github/metadata_mining/middle_dir:/middle_dir \
  metadmin \
  conda run -n metadmin_env python /app/scripts/parse_lat_lon_from_metadata.py \
  | grep '^OUTPUT:' \
  | cut -f1-5 \
  | tr '\t' ' ' \
  | sed 's/  */ /g' \
  | sed 's/ *$//' \
  > ~/github/metadata_mining/middle_dir/sample.coordinates.reparsed.filtered_fresh
```
 
