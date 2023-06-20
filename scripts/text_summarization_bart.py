#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:32:43 2023

@author: dgaio
"""

from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize the tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# The text to be summarized
text_to_summarize = """
The Amazon rainforest, alternatively, the Amazon Jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories.
"""

# Prepare the text source
inputs = tokenizer([text_to_summarize], max_length=1024, return_tensors='pt')

# Generate the summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

print(summary)

len(text_to_summarize)
len(summary[0].split())

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

text_to_summarize = '''
>SRS1228264
sample_center_name=
sample_alias=B513M304x
sample_TAXON_ID=410658
sample_SCIENTIFIC_NAME=soil metagenome
sample_XREF_LINK=bioproject: 283223)
sample_isolation_source=Agricultural field soil, top 10 cm
sample_collection_date=May-2013
sample_geo_loc_name=USA: Southern Minnesota
sample_lat_lon=45.01 N 96.15 W
sample_samp_collect_device=0.75-in soil core to 10 cm depth
sample_samp_size=0.5 g extracted from 3 homogenized cores
sample_Plot=M304
sample_Timepoint=B513
sample_Subsample=x
sample_Amendment treatment=Winter rye (Secale cereale)
sample_Field sample code=B513M304x
sample_BioSampleModel=Metagenome or environmental
experiments=SRX1507173
study=SRP067882
study_STUDY_TITLE=Organic agricultural field soil Raw sequence reads
study_STUDY_TYPE=
study_STUDY_ABSTRACT=Study investigated effects of cover crops and organic fertilizers on bacterial community structure in organically managed agricultural field soils in Minnesota
experiment=SRX1507173
experiment_external_id_BioProject=PRJNA283223
experiment_DESIGN_DESCRIPTION=16S rDNA V5-V6 hypervariable region
experiment_external_id_BioSample=SAMN04360453
experiment_LIBRARY_NAME=B513M304x_cerc
experiment_LIBRARY_STRATEGY=AMPLICON
experiment_LIBRARY_SOURCE=METAGENOMIC
experiment_LIBRARY_SELECTION=PCR
experiment_LIBRARY_LAYOUT_PAIRED= :: 
experiment_instrument=Illumina MiSeq
experiment_alignment_software=mothur v. 1.33.3
runs=SRR3062996
'''

