#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:35:46 2024

@author: dgaio
"""


# to test instructing gpt to output json format responses 
# conclusion: ca 4x as expensive output token -wise 

import json
import openai


content_string = '''
'sample_ID=ERS2678531': '>ERS2678531
sample_alias=SAMEA4858692
sample_TITLE=U11S
sample_TAXON_ID=1869227
sample_SCIENTIFIC_NAME=bacterium
sample_Alias=U11S
sample_ENA checklist=ERC000024
sample_INSDC center alias=Instituto Antofagasta
sample_INSDC center name=Instituto Antofagasta
sample_INSDC first public=2018-10-24T17:02:17Z
sample_INSDC last update=2018-08-24T19:25:51Z
sample_INSDC status=public
sample_SRA accession=ERS2678531
sample_Sample Name=ERS2678531
sample_Title=Sediment and water
sample_collection date=2017
sample_environment (biome)=river
sample_environment (feature)=river
sample_environment (material)=water
sample_geographic location (country and/or sea)=Chile
sample_geographic location (depth)=1
sample_geographic location (latitude)=-21.4299600
sample_geographic location (longitude)=-70.0587400
sample_investigation type=metagenome
sample_project name=Connectivity of Bacterial Communities-Case of the Loa River in the Atacama Desert
sample_sequencing method=Illumina Miseq
sample_water environmental package=water
study=ERP110559
study_STUDY_TITLE=Connectivity of Bacterial Communities—Case of the Loa River in the Atacama Desert
study_STUDY_ABSTRACT=The Loa River is an exceptional watercourse within the Atacama Desert; it crosses ~440km from East to West and discharges its waters into the Pacific Ocean. However, the roleof this fluvial ecosystem in the dispersal of microorganisms has not yet been studied indetail. In this study, we analysed bacterial communities in the Loa River, using V4 regionsequencing of the 16S rRNA gene from 16 samples of water and surface sediments takenfrom 8 sampling sites along the river. In addition, we used network analysis to identifyinterconnected river microbial assemblages. An average of 2482 Operational TaxonomicUnits (OTUs) were obtained, whose diversity decreased from the headstream down andtended to be greater in sediment than in water. The structure of the bacterial communitywas related along the river, clustering by site and based on substrate type (water orsediment). The communities were dominated by a large variety of Proteobacteria,Bacteroidetes, Cyanobacteria, Planctomycetes, Actinobacteria, and Firmicutes phyla.Further, less abundant communities contained many unique taxa with a high level ofmetabolic diversity. Shifts in community structure were related to the influence of salinity,pH, chlorophyll, and oxygen saturation. The proportions of common taxa along the riverindicated that there is strong connectivity between the bacterial communities upstream anddownstream. Thus, we propose the Loa River is a possible &quot;extreme local reservoir&quot; with agreat ecological importance, emphasising the need for planning its conservation.
study_STUDY_DESCRIPTION=The Loa River is an exceptional watercourse within the Atacama Desert; it crosses ~440km from East to West and discharges its waters into the Pacific Ocean. However, the roleof this fluvial ecosystem in the dispersal of microorganisms has not yet been studied indetail. In this study, we analysed bacterial communities in the Loa River, using V4 regionsequencing of the 16S rRNA gene from 16 samples of water and surface sediments takenfrom 8 sampling sites along the river. In addition, we used network analysis to identifyinterconnected river microbial assemblages. An average of 2482 Operational TaxonomicUnits (OTUs) were obtained, whose diversity decreased from the headstream down andtended to be greater in sediment than in water. The structure of the bacterial communitywas related along the river, clustering by site and based on substrate type (water orsediment). The communities were dominated by a large variety of Proteobacteria,Bacteroidetes, Cyanobacteria, Planctomycetes, Actinobacteria, and Firmicutes phyla.Further, less abundant communities contained many unique taxa with a high level ofmetabolic diversity. Shifts in community structure were related to the influence of salinity,pH, chlorophyll, and oxygen saturation. The proportions of common taxa along the riverindicated that there is strong connectivity between the bacterial communities upstream anddownstream. Thus, we propose the Loa River is a possible &quot;extreme local reservoir&quot; with agreat ecological importance, emphasising the need for planning its conservation.
study_ENA-FIRST-PUBLIC=2018-10-24
study_ENA-LAST-UPDATE=2018-08-24'
~~~
'sample_ID=SRS2066445': '>SRS2066445
sample_alias=CG_RF_114
sample_TITLE=coral-associated prokaryotic community
sample_TAXON_ID=408172
sample_SCIENTIFIC_NAME=marine metagenome
sample_DESCRIPTION=CoralGardens_ReefFlat
sample_XREF_LINK=bioproject: 380169)
sample_host=Pocillopora damicornis
sample_isolation_source=coral tissue
sample_collection_date=2012
sample_geo_loc_name=Australia: Heron Island
sample_lat_lon=14.61628 S 145.63683 E
sample_source_material_id=CG_RF_114
sample_BioSampleModel=Metagenome or environmental
study=SRP102380
study_STUDY_TITLE=marine metagenome Raw sequence reads
study_STUDY_ABSTRACT=Adaptation to reef habitats through selection on the coral animal and its associated microbiome'
'''

api_key_path = '/Users/dgaio/my_api_key'
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()
    
system_prompt = '''
Based on the metadata texts provided, please generate a plain JSON response with the following details:
- "sample_id": The ID of the sample.
- "origin_type": Guess where the metagenomic sample each metadata text is based on, comes from. Your choices are: 'animal' (incl. human), 'plant', 'water', 'soil', 'other'. Provide strictly one-word answer.
- "geographical_location": Guess where, geographically, the sample was collected from (including country).
- "sample_detail": Give detail about the sample origin in maximum 3 words: specify which host and what part of the host for 'animal' or 'plant', specify which type (e.g., lake, brine, sea) for 'water', specify which type (e.g., agricultural, desert, forest) for 'soil', specify which type (e.g., urban, laboratory) for 'other'. When info is not available, write 'NA'.

Please provide the response in plain JSON format without any additional code block or markdown formatting.
'''

    
response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": system_prompt  # use the loaded system prompt
            },
            {
                "role": "user",
                "content": content_string
            }
        ],
        temperature=1.00,
        max_tokens=4096,
        top_p=0.75,
        frequency_penalty=0.25,
        presence_penalty=1.5
    )
    

# Assume `response` is the object you received from OpenAI
response_content = response['choices'][0]['message']['content']

print(response_content)







##############################


# Using function calling feature of OpenAI (example):

student_custom_functions = [
    {
        'name': 'extract_student_info',
        'description': 'Get the student information from the body of the input text',
        'parameters': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                    'description': 'Name of the person'
                },
                'major': {
                    'type': 'string',
                    'description': 'Major subject.'
                },
                'school': {
                    'type': 'string',
                    'description': 'The university name.'
                },
                'grades': {
                    'type': 'integer',
                    'description': 'GPA of the student.'
                },
                'club': {
                    'type': 'string',
                    'description': 'School club for extracurricular activities. '
                }
                
            }
        }
    }
]


student_1_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."
student_2_description="Ravi Patel is a sophomore majoring in computer science at the University of Michigan. He is South Asian Indian American and has a 3.7 GPA. Ravi is an active member of the university's Chess Club and the South Asian Student Association. He hopes to pursue a career in software engineering after graduating."



student_description = [student_1_description,student_2_description]
for sample in student_description:
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = [{'role': 'user', 'content': sample}],
        functions = student_custom_functions,
        function_call = 'auto'
    )

    # Loading the response as a JSON object
    json_response = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    print(json_response)



##############################


# In this way below, we run 1 sample at a time. Works well, but it sevred me to understand that text in functions is indeed calculated as prompt tokens. 



sample_custom_functions = [
    {
        'name': 'extract_sample_info',
        'description': ''''From the metadata texts below, gather the following values. Separate samples are separated by '~~~'. Fill with NA when value can't be given. ''',
        'parameters': {
            'type': 'object',
            'properties': {
                '1': {
                    'type': 'string',
                    'description': 'Extract the sample ID.'
                },
                '2': {
                    'type': 'string',
                    'description': '''Guess its origin. Your choices are: 'animal' (incl. human), 'plant', 'water', 'soil', 'other'. Give strictly 1-word answer.'''
                },
                '3': {
                    'type': 'string',
                    'description': 'Guess where, geographically, the sample was collected from (including country).'
                },
                '4': {
                    'type': 'string',
                    'description': ''''Give detail about the sample origin in maximum 3 words: If from an 'animal' or a 'plant', specify which host and what part of the host. If from 'water' specify which (e.g.: river, brine, sea, waste water, etc). If from 'soil' specify which (e.g.: agricultural, desert, forest, etc). If from 'other' specify which (e.g.: urban, laboratory, feed/food, fungus, air, etc).'''
                }
                
            }
        }
    }
]



sample_1_description = '''
'sample_ID=ERS2678531': '>ERS2678531
sample_alias=SAMEA4858692
sample_TITLE=U11S
sample_TAXON_ID=1869227
sample_SCIENTIFIC_NAME=bacterium
sample_Alias=U11S
sample_ENA checklist=ERC000024
sample_INSDC center alias=Instituto Antofagasta
sample_INSDC center name=Instituto Antofagasta
sample_INSDC first public=2018-10-24T17:02:17Z
sample_INSDC last update=2018-08-24T19:25:51Z
sample_INSDC status=public
sample_SRA accession=ERS2678531
sample_Sample Name=ERS2678531
sample_Title=Sediment and water
sample_collection date=2017
sample_environment (biome)=river
sample_environment (feature)=river
sample_environment (material)=water
sample_geographic location (country and/or sea)=Chile
sample_geographic location (depth)=1
sample_geographic location (latitude)=-21.4299600
sample_geographic location (longitude)=-70.0587400
sample_investigation type=metagenome
sample_project name=Connectivity of Bacterial Communities-Case of the Loa River in the Atacama Desert
sample_sequencing method=Illumina Miseq
sample_water environmental package=water
study=ERP110559
study_STUDY_TITLE=Connectivity of Bacterial Communities—Case of the Loa River in the Atacama Desert
study_STUDY_ABSTRACT=The Loa River is an exceptional watercourse within the Atacama Desert; it crosses ~440km from East to West and discharges its waters into the Pacific Ocean. However, the roleof this fluvial ecosystem in the dispersal of microorganisms has not yet been studied indetail. In this study, we analysed bacterial communities in the Loa River, using V4 regionsequencing of the 16S rRNA gene from 16 samples of water and surface sediments takenfrom 8 sampling sites along the river. In addition, we used network analysis to identifyinterconnected river microbial assemblages. An average of 2482 Operational TaxonomicUnits (OTUs) were obtained, whose diversity decreased from the headstream down andtended to be greater in sediment than in water. The structure of the bacterial communitywas related along the river, clustering by site and based on substrate type (water orsediment). The communities were dominated by a large variety of Proteobacteria,Bacteroidetes, Cyanobacteria, Planctomycetes, Actinobacteria, and Firmicutes phyla.Further, less abundant communities contained many unique taxa with a high level ofmetabolic diversity. Shifts in community structure were related to the influence of salinity,pH, chlorophyll, and oxygen saturation. The proportions of common taxa along the riverindicated that there is strong connectivity between the bacterial communities upstream anddownstream. Thus, we propose the Loa River is a possible &quot;extreme local reservoir&quot; with agreat ecological importance, emphasising the need for planning its conservation.
study_STUDY_DESCRIPTION=The Loa River is an exceptional watercourse within the Atacama Desert; it crosses ~440km from East to West and discharges its waters into the Pacific Ocean. However, the roleof this fluvial ecosystem in the dispersal of microorganisms has not yet been studied indetail. In this study, we analysed bacterial communities in the Loa River, using V4 regionsequencing of the 16S rRNA gene from 16 samples of water and surface sediments takenfrom 8 sampling sites along the river. In addition, we used network analysis to identifyinterconnected river microbial assemblages. An average of 2482 Operational TaxonomicUnits (OTUs) were obtained, whose diversity decreased from the headstream down andtended to be greater in sediment than in water. The structure of the bacterial communitywas related along the river, clustering by site and based on substrate type (water orsediment). The communities were dominated by a large variety of Proteobacteria,Bacteroidetes, Cyanobacteria, Planctomycetes, Actinobacteria, and Firmicutes phyla.Further, less abundant communities contained many unique taxa with a high level ofmetabolic diversity. Shifts in community structure were related to the influence of salinity,pH, chlorophyll, and oxygen saturation. The proportions of common taxa along the riverindicated that there is strong connectivity between the bacterial communities upstream anddownstream. Thus, we propose the Loa River is a possible &quot;extreme local reservoir&quot; with agreat ecological importance, emphasising the need for planning its conservation.
study_ENA-FIRST-PUBLIC=2018-10-24
study_ENA-LAST-UPDATE=2018-08-24'
'''


sample_2_description= '''
'sample_ID=SRS2066445': '>SRS2066445
sample_alias=CG_RF_114
sample_TITLE=coral-associated prokaryotic community
sample_TAXON_ID=408172
sample_SCIENTIFIC_NAME=marine metagenome
sample_DESCRIPTION=CoralGardens_ReefFlat
sample_XREF_LINK=bioproject: 380169)
sample_host=Pocillopora damicornis
sample_isolation_source=coral tissue
sample_collection_date=2012
sample_geo_loc_name=Australia: Heron Island
sample_lat_lon=14.61628 S 145.63683 E
sample_source_material_id=CG_RF_114
sample_BioSampleModel=Metagenome or environmental
study=SRP102380
study_STUDY_TITLE=marine metagenome Raw sequence reads
study_STUDY_ABSTRACT=Adaptation to reef habitats through selection on the coral animal and its associated microbiome'
'''



sample_description = [sample_1_description,sample_2_description]
for sample in sample_description:
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo-1106',
        
        messages = [{'role': 'user', 'content': sample}],
        
        functions = sample_custom_functions,
        function_call = 'auto',
        temperature=1.00,
        max_tokens=4096,
        top_p=0.75,
        frequency_penalty=0.25,
        presence_penalty=1.5
    )

    # Loading the response as a JSON object
    json_response = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    print(response)
    

##############################

# This below shows that binding two samples texts together doesn t give a good output. Only 1 out of 2 samples recalled. 

combined_sample_description = sample_1_description + '~~~' + sample_2_description

# Make a single API call with the combined description
response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo-1106',
    messages = [{'role': 'user', 'content': combined_sample_description}],
    functions = sample_custom_functions,
    function_call = 'auto',
    temperature=1.00,
    max_tokens=4096,
    top_p=0.75,
    frequency_penalty=0.25,
    presence_penalty=1.5
)

print(response)

##############################



# thiw is what happens when main info is given on system prompt, and functions only describe values allocations: 
    

sample_custom_functions = [
    {
        'name': 'extract_sample_info',
        'description': 'Get the sample information from the body of the input text',
        'parameters': {
            'type': 'object',
            'properties': {
                '1': {
                    'type': 'string',
                    'description': 'sample ID'
                },
                '2': {
                    'type': 'string',
                    'description': 'sample origin'
                },
                '3': {
                    'type': 'string',
                    'description': 'geo location'
                },
                '4': {
                    'type': 'string',
                    'description': 'sample details'
                }
                
            }
        }
    }
]




system_prompt = '''
Based on the metadata texts below:
- Guess where the metagenomic sample each metadata text is based on, comes from. Your choices are: 'animal' (incl. human), 'plant', 'water', 'soil', 'other'. Give strictly 1-word answer for each sample ID.
- Guess where, geographically, the sample was collected from (including country). 
- Give detail about the sample origin in maximum 3 words: If from an 'animal' or a 'plant', specify which host and what part of the host. If from 'water' specify which (e.g.: river, brine, sea, waste water, etc). If from 'soil' specify which (e.g.: agricultural, desert, forest, etc). If from 'other' specify which (e.g.: urban, laboratory, feed/food, fungus, air, etc). 
When info is not available, write 'NA'. 
'''


response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo-1106',
    

    messages = [{'role': 'user', 'content': content_string}],
    
    functions = sample_custom_functions,
    function_call = 'auto',
    temperature=1.00,
    max_tokens=4096,
    top_p=0.75,
    frequency_penalty=0.25,
    presence_penalty=1.5
)


print(response)

content_str = response['choices'][0]['message']['function_call']['arguments']
json_response = json.loads(content_str)
print(json_response)





