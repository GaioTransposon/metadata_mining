#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:31:59 2023

@author: dgaio
"""

import random
import string
import time
import string
import gzip
import re



start = time.time()


def compress_text(text):
    compressed_data = gzip.compress(text.encode('utf-8'))
    return compressed_data

def decompress_text(compressed_data):
    decompressed_text = gzip.decompress(compressed_data).decode('utf-8')
    return decompressed_text

# Example usage
original_text = "Streptococcus suis is a major porcine and zoonotic pathogen responsible for significant economic losses in the pig industry and an increasing number of human cases. Multiple isolates of S. suis show marked genomic diversity. Here, we report the analysis of whole genome sequences of nine pig isolates that caused disease typical of S. suis and had phenotypic characteristics of S. suis, but their genomes were divergent from those of many other S. suis isolates. Comparison of protein sequences predicted from divergent genomes with those from normal S. suis reduced the size of core genome from 793 to only 397 genes. Divergence was clear if phylogenetic analysis was performed on reduced core genes and MLST alleles. Phylogenies based on certain other genes (16S rRNA, sodA, recN, and cpn60) did not show divergence for all isolates, suggesting recombination between some divergent isolates with normal S. suis for these genes. Indeed, there is evidence of recent recombination between the divergent and normal S. suis genomes for 249 of 397 core genes. In addition, phylogenetic analysis based on the 16S rRNA gene and 132 genes that were conserved between the divergent isolates and representatives of the broader Streptococcus genus showed that divergent isolates were more closely related to S. suis. Six out of nine divergent isolates possessed a S. suis-like capsule region with variation in capsular gene sequences but the remaining three did not have a discrete capsule locus. The majority (40/70), of virulence-associated genes in normal S. suis were present in the divergent genomes. Overall, the divergent isolates extend the current diversity of S. suis species but the phenotypic similarities and the large amount of gene exchange with normal S. suis gives insufficient evidence to assign these isolates to a new species or subspecies. Further, sampling and whole genome analysis of more isolates is warranted to understand the diversity of the species"
compressed_data = compress_text(original_text)

print("Original Text:")
print(original_text)

print("\nCompressed Data:")
print(compressed_data)

decompressed_text = decompress_text(compressed_data)

print("\nDecompressed Text:")
print(decompressed_text)

print("\nNumber of words in original text:", len(original_text.split()))
print("Number of words in compressed text:", len(compressed_data.split()))





end = time.time()

print("Running time: ", end-start)	



# take text

# compress it 

# feed it to AI and ask to: 
    # 1. decompress it with function
    # 2. make a summary 
    # 3. extract keywords
    # 4. predict biome
    # 5. compress these three and return them
    
# decompress AI output 


####################################################
####################################################





start = time.time()

def generate_key(sentence):
    key = {}
    words = sentence.split()
    for i in range(len(words) - 1):
        phrase = words[i] + " " + words[i+1]
        if phrase not in key:
            key[phrase] = f"<Word{len(key) + 1}>"
    return key

def transform_sentence(sentence, key):
    transformed_sentence = sentence
    for phrase, replacement in key.items():
        transformed_sentence = transformed_sentence.replace(phrase, replacement)
    return transformed_sentence

def reproduce_sentence(transformed_sentence, key):
    reproduced_sentence = transformed_sentence
    for phrase, replacement in key.items():
        reproduced_sentence = reproduced_sentence.replace(replacement, phrase)
    return reproduced_sentence

# Original sentence
original_sentence = "Streptococcus suis is a major porcine and zoonotic pathogen responsible for significant economic losses in the pig industry and an increasing number of human cases and an other thing to add."

# Generate the key
key = generate_key(original_sentence)

# Transform the sentence
transformed_sentence = transform_sentence(original_sentence, key)

# Output the transformed sentence and the key
print("Transformed Sentence:")
print(transformed_sentence)

print("\nKey:")
for phrase, replacement in key.items():
    print(f"{phrase} = {replacement}")

# Reproduce the original sentence
reproduced_sentence = reproduce_sentence(transformed_sentence, key)

print("\nReproduced Sentence:")
print(reproduced_sentence)


end = time.time()

print("Running time: ", end-start)	


####################################################
####################################################


# 1. take metadata
# 2. generate key
# 3. transform it (ori --> encrypted)

# 4. feed it to AI, providing key
# 5. get AI to transform it (encrypted --> ori)
# # 6. make summary 
# # 7. extract keywords
# # 8. predict biome (with probability score)
# # 9. (evt) predict geographic info
# 10. use key to transform it (ori --> encrypted)

# 11. transform it (encrypted --> ori)




def generate_key(sentence):
    key = {}
    words = sentence.split()
    for i in range(len(words) - 1):
        phrase = words[i] + " " + words[i+1]
        if phrase not in key:
            key[phrase] = f"<Word{len(key) + 1}>"
    return key

def transform_sentence(sentence, key):
    transformed_sentence = sentence
    for phrase, replacement in key.items():
        transformed_sentence = transformed_sentence.replace(phrase, replacement)
    return transformed_sentence

# Function to read text from a .txt file
def read_text_from_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

# Function to write text to a .txt file
def write_text_to_file(text, file_path):
    with open(file_path, "w") as file:
        file.write(text)

# Example usage
input_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/text.txt"
output_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/text_out.txt"

# Read the text from the input file
original_text = read_text_from_file(input_file_path)

# Generate the key and transform the sentence
key = generate_key(original_text)
transformed_sentence = transform_sentence(original_text, key)

# Write the transformed sentence to the output file
write_text_to_file(transformed_sentence, output_file_path)

# Print the key and transformed sentence
print("Key:")
for phrase, replacement in key.items():
    print(f"{phrase} = {replacement}")

print("\nTransformed Sentence:")
print(transformed_sentence)
print(len(original_text.split()))
print(len(transformed_sentence.split()))








original_sentence = "Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens."

# Generate the key
key = generate_key(original_sentence)

# Transform the sentence
transformed_sentence = transform_sentence(original_sentence, key)

# Output the transformed sentence and the key
print("Transformed Sentence:")
print(transformed_sentence)

len(original_sentence.split())
len(transformed_sentence.split())


