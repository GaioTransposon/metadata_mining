#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:19:53 2023

@author: dgaio
"""
import openai
import os

# Set up the OpenAI API key
openai.api_key = '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'

# Send a completion request
response = openai.Completion.create(
  engine="text-davinci-001", # Choose the GPT-3.5 engine
  prompt="From the following text, do these tasks: 1. make a summary in less than 100, 2. extract 3 keywords, 3. extract geographic location of sample, 4. make a guess of the sample's origin between animal, soil, water, or plant. Here is the text: " + text_to_summarize_1,
  max_tokens=200
)



print(response.choices[0].text.strip())




text_to_summarize_1 = '''
>SRS7137346
sample_center_name=
sample_alias=hand_40_2
sample_TAXON_ID=646099
sample_SCIENTIFIC_NAME=human metagenome
sample_XREF_LINK=bioproject: 650212)
sample_isolate=Skin_microbes_2
sample_collection_date=01-Mar-2017
sample_env_broad_scale=skin_microbiome
sample_env_local_scale=skin
sample_env_medium=skin
sample_geo_loc_name=Korea: Seoul
sample_host=Homo sapiens
sample_isol_growth_condt=missing
sample_lat_lon=37.5665 N 126.9780 E
sample_BioSampleModel=MIMARKS.specimen
sample_BioSampleModel=MIGS/MIMS/MIMARKS.human-skin
experiments=SRX8876758
study=SRP275714
study_STUDY_TITLE=Human skin microbiota and related microbial functions depending on the ages
study_STUDY_TYPE=
study_STUDY_ABSTRACT=This project intended to very the differentially abundant microbial taxa and their related functional features depending on the ages and skin locations.
experiment=SRX8876758
experiment_DESIGN_DESCRIPTION=Prokaryotic population from human skin
experiment_LIBRARY_NAME=hand_40_2
experiment_LIBRARY_STRATEGY=AMPLICON
experiment_LIBRARY_SOURCE=METAGENOMIC
experiment_LIBRARY_SELECTION=unspecified
experiment_LIBRARY_LAYOUT_PAIRED= :: 
experiment_instrument=Illumina MiSeq
runs=SRR12378220
'''
