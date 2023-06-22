#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:59:51 2023

@author: dgaio
"""


from transformers import pipeline

nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "The event will take place at Mirissa at coordinates 37.7749° N, 122.4194° W."

ner_results = nlp(text)

for result in ner_results:
    if result['entity'] == 'I-LOC':
        print(f"Entity: {result['word']}, Score: {result['score']}")
    
    
    
# These models often use a technique called "subword tokenization" to deal with words not seen during training. In your example, the model split the word "Mirissa" into two subword tokens: "Mir" and "##issa". The "##" in front of "issa" indicates that this subword is the continuation of the previous token.
# So you need to aggregate them. 
def aggregate_entities(ner_results):
    aggregated_entities = []
    current_entity_parts = []
    current_entity_label = None

    for result in ner_results:
        # Check if this result is part of the current entity
        if result['entity'] == current_entity_label and result['word'].startswith('##'):
            current_entity_parts.append(result['word'][2:])
        else:
            # If it's a new entity, add the current entity to the aggregated_entities list
            if current_entity_parts:
                aggregated_entities.append({
                    'entity': current_entity_label,
                    'word': ''.join(current_entity_parts)
                })

            # Start a new entity
            current_entity_parts = [result['word'].replace('##', '')]
            current_entity_label = result['entity']

    # Add the last entity to the list
    if current_entity_parts:
        aggregated_entities.append({
            'entity': current_entity_label,
            'word': ''.join(current_entity_parts)
        })

    return aggregated_entities

print(aggregate_entities(ner_results))




# coordinates
# cordinates
# coordinate
# cordinate
# coord
# lat_lon
# lon_lat
# lon/lat
# lat/lon

