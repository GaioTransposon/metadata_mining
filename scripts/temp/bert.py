#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:20:37 2023

@author: dgaio
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizer


# from here: https://huggingface.co/models
model_name = "gpt2"  # Replace with the name or path of the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_name)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

## alternatve:
## tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')





text = "Bacterial transformation is of great importance both clinically, as it can faciiltate the acquisition of antibiotic resistance and the evasion of vaccines, and biologically, as it plays a a critical role in affecting bacterial population structure and speciation. This analysis is designed to characterise the constraints on genetic exchange through bacterial transformation. A recipient genotype, Streptococcus pneumoniae R6I-20d, was transformed with the same rifampicin resistance mutation isolated in different streptococcal donor species, to identify the important aspects of interspecies variation that limited recombination. "


tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

outputs = model(**tokens)
predicted_labels = outputs.logits.argmax(dim=1)










from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Define the input text
input_text = """
    By similarly performing experiments with different S. pneumoniae donors and mutant recipient strains, we have 
identified limitations to the rate of transfer of multiple types of antibiotic resistance marker, and the moderation of transformation rate by different aspects of 
the machinery involved in the process.
"""

# Tokenize the input text
input_ids = tokenizer.encode(input_text, max_length=1024, truncation=True, return_tensors="pt")

# Generate the summary
summary_ids = model.generate(input_ids, num_beams=4, max_length=150, early_stopping=True)
summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

print("Summary:")
print(summary)











from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Define the input text
input_text = """
    By similarly performing experiments with different S. pneumoniae donors and mutant recipient strains, we have 
identified limitations to the rate of transfer of multiple types of antibiotic resistance marker, and the moderation of transformation rate by different aspects of 
the machinery involved in the process.
"""

# Tokenize the input text
tokens = tokenizer.tokenize(input_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([input_ids])

# Obtain the keyword embeddings
with torch.no_grad():
    outputs = model(input_ids)
    keyword_embeddings = outputs.last_hidden_state.squeeze(0)

# Perform keyword extraction (example: top 5 keywords)
num_keywords = 5
keyword_indices = keyword_embeddings.sum(dim=1).topk(num_keywords).indices.tolist()

# Retrieve the keywords from the tokenizer
keywords = [tokens[idx] for idx in keyword_indices]

print("Keywords:")
print(keywords)



from summa import keywords

# Define the input text
input_text = """
    By similarly performing experiments with different S. pneumoniae donors and mutant recipient strains, we have 
identified limitations to the rate of transfer of multiple types of antibiotic resistance marker, and the moderation of transformation rate by different aspects of 
the machinery involved in the process.
"""

# Extract keywords using TextRank
text_keywords = keywords.keywords(input_text)

# Split the extracted keywords into a list
keyword_list = text_keywords.split('\n')

print("Keywords:")
print(keyword_list)





from keybert import KeyBERT

# Define the input text
input_text = """
    Your input text goes here.
"""

# Initialize the KeyBERT model
model = KeyBERT()

# Extract keywords
keywords = model.extract_keywords(input_text)

# Get the top N keywords
top_keywords = [kw for kw, _ in keywords]

print("Keywords:")
print(top_keywords)








from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))




metadata_text = """
>ERS3720124
sample_center_name=
sample_alias=SAMEA5931644
sample_TAXON_ID=1313
sample_SCIENTIFIC_NAME=Streptococcus pneumoniae
sample_ArrayExpress-SPECIES=Streptococcus pneumoniae
sample_ArrayExpress-STRAIN_OR_LINE=R6I-20d
sample_ENA first public=2020-11-10
sample_ENA last update=2019-09-04
sample_External Id=SAMEA5931644
sample_INSDC center alias=SC
sample_INSDC center name=Wellcome Sanger Institute
sample_INSDC first public=2020-11-10T17:01:29Z
sample_INSDC last update=2019-09-04T08:52:08Z
sample_INSDC status=public
sample_SRA accession=ERS3720124
sample_Sample Name=2b70c308-cee8-11e9-88a2-68b599768938
sample_Submitter Id=2b70c308-cee8-11e9-88a2-68b599768938
sample_common name=Streptococcus pneumoniae
sample_subject id=5940STDY8158631
sample_title=R6I20d-SK23_C_3_i
experiments=ERX4694172
study=ERP117105
study_STUDY_TITLE=Characterising_streptococcal_transformation_through_sequencing_in_vitro_recombinants
study_STUDY_TYPE=
study_STUDY_ABSTRACT=Bacterial transformation is of great importance both clinically, as it can faciiltate the acquisition of antibiotic resistance and the evasion of 
vaccines, and biologically, as it plays a a critical role in affecting bacterial population structure and speciation. This analysis is designed to characterise the 
constraints on genetic exchange through bacterial transformation. A recipient genotype, Streptococcus pneumoniae R6I-20d, was transformed with the same rifampicin 
resistance mutation isolated in different streptococcal donor species, to identify the important aspects of interspecies variation that limited recombination. 
Similarly, other resistance markers in S. pneumoniae were also transformed into R6I-20d, to identify how the properties of homologous recombination changed with the 
structural properties of the resistance determinant. Finally, mutant derivatives of R6I-20d were transformed with the same resistance markers, to identify which 
aspects of the machinery controlled different properties of the recombination process. By better understanding the transformation process, we can greatly enhance our 
understanding of bacterial evolution and epidemiology.
study_STUDY_DESCRIPTION=Bacterial transformation is of great importance both clinically, as it can faciiltate the acquisition of antibiotic resistance and the evasion 
of vaccines, and biologically, as it plays a a critical role in affecting bacterial population structure and speciation. To characterise the constraints on horizontal 
DNA transfer through this process, we have conducted a series of experiments using different donors and recipients to generate recombinant bacteria in the laboratory. 
By quantifying the rates of exchange between donors of different species into a Streptococccus pneumoniae recipient, we have detected a limitation on the 
transformation rate by interspecies genetic differences. By similarly performing experiments with different S. pneumoniae donors and mutant recipient strains, we have 
identified limitations to the rate of transfer of multiple types of antibiotic resistance marker, and the moderation of transformation rate by different aspects of 
the machinery involved in the process. By sequencing these transformant bacteria, we will be able to characterise exactly what types of genetic variation limit 
genetic exchange between species, and improve our understanding of homologous recombination in bacteria. This data is part of a pre-publication release. For 
information on the proper use of pre-publication data shared by the Wellcome Trust Sanger Institute (including details of any publication moratoria), please see 
http://www.sanger.ac.uk/datasharing/
study_ENA-FIRST-PUBLIC=2020-11-10
study_STUDY_ATTRIBUTE=      
study_ENA-LAST-UPDATE=2019-09-03
study_STUDY_ATTRIBUTE=      
study_STUDY_ATTRIBUTES=    
experiment=ERX4694172
experiment_external_id_BioProject=PRJEB34230
experiment_DESIGN_DESCRIPTION=Illumina sequencing of library DN538835C:H7, constructed from sample accession ERS3720124 for study accession ERP117105.  This is part 
of an Illumina multiplexed sequencing run (31585_6).  This submission includes reads tagged with the sequence AGACACTA.
experiment_external_id_BioSample=SAMEA5931644
experiment_LIBRARY_NAME=DN538835C:H7
experiment_LIBRARY_STRATEGY=WGS
experiment_LIBRARY_SOURCE=GENOMIC
experiment_LIBRARY_SELECTION=RANDOM
experiment_LIBRARY_LAYOUT_PAIRED= :: NOMINAL_LENGTH=463,NOMINAL_SDEV=124
experiment_LIBRARY_CONSTRUCTION_PROTOCOL=Standard
experiment_instrument=HiSeq X Ten
runs=ERR4824188
"""


from transformers import BertTokenizer, BertForSequenceClassification


# Define the number of classes (origins) and label mapping
num_classes = 3
label_mapping = {
    0: "water",
    1: "soil",
    2: "animal"
}



# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Preprocess the metadata text
metadata_text = """
    # Your metadata text here
"""
input_ids = tokenizer.encode(metadata_text, add_special_tokens=True)
input_ids = input_ids[:512]  # Truncate if necessary
input_ids = torch.tensor([input_ids])

# Make the prediction
outputs = model(input_ids)
predicted_label = torch.argmax(outputs.logits, dim=1).item()

# Map the predicted label to the corresponding origin
origin = label_mapping[predicted_label]
print("Predicted Origin:", origin)


















import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataframe and extract necessary columns
file_path = os.path.expanduser('~/github/metadata_mining/middle_dir/pubmed_articles_info_for_training.csv')
data = pd.read_csv(file_path).dropna(subset=['confirmed_biome'])
labels = data['confirmed_biome']
texts = data['abstract']

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Tokenize and encode the texts
max_length = 200  # Maximum sequence length supported by BERT
tokenized_texts = []
for text in texts:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + [tokens[-1]]  # Truncate to max_length - 1 tokens
    tokenized_texts.append(tokens)

# Pad the tokenized sequences to have equal length
padded_texts = torch.tensor([tokens + [0] * (max_length - len(tokens)) for tokens in tokenized_texts])

# Obtain BERT embeddings
with torch.no_grad():
    embeddings = model(padded_texts)[0][:, 0, :].numpy()  # Extract the first token's embeddings

# Apply dimensionality reduction using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Create a scatter plot for visualization
plt.figure(figsize=(10, 8))
for label in set(labels):
    indices = labels == label
    plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label)
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('BERT Embeddings Visualization')
plt.show()




