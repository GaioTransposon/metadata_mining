#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:08:44 2023

@author: dgaio
"""


# PART 2: 
# goes through raw metadata files and replaces labels at each occurrence (regex flexible)
# with their respectective description. 
# saves it with the same name followed by _clean before.txt 






def create_regex_pattern(label):
    # Extract the prefix and the digits from the label
    match = re.match(r'([a-zA-Z]+)_?(\d+)', label)
    
    if not match:
        raise ValueError(f"Unexpected label format: {label}")

    prefix, digits = match.groups()
    
    # Construct a regex pattern
    pattern = prefix + r'\D+' + digits
    return pattern






def process_metadata(samples, label_info_dict, base_dir):
    shuffled_samples = samples.sample(frac=1).reset_index(drop=True)  # Removed seed as it wasn't provided

    processed_samples_count = 0
    processed_samples_list = []
    endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
    metadata_dict = {}
    
    
    for _, row in shuffled_samples.iterrows():
        # Extract the last three digits and construct the directory path
        sub_dir = f"dir_{row['sample'][-3:]}"
        sample_file = os.path.join(base_dir, sub_dir, row['sample'] + '.txt')
        
        with open(sample_file, 'r') as f:
            metadata = f.read()
            for label, joint_info in label_info_dict.items():
                pattern = create_regex_pattern(label)
                metadata = re.sub(pattern, joint_info, metadata, flags=re.IGNORECASE)

        cleaned_metadata_lines = []
        for line in metadata.splitlines():
            stripped_line = line.strip()
            should_keep = True
            if stripped_line.lower().startswith(("experiment", "run", ">")):
                should_keep = False
            else:
                for ending in endings_to_remove:
                    if re.search(ending, stripped_line, re.IGNORECASE):
                        print(f"Rejected line (ends with {ending}): {stripped_line}")
                        should_keep = False
                        break
            if should_keep:
                cleaned_metadata_lines.append(stripped_line)

        cleaned_metadata = "\n".join(cleaned_metadata_lines)
        metadata_dict[row['sample']] = cleaned_metadata
            
        # Save cleaned metadata to file
        clean_file_name = sample_file.replace('.txt', '_clean.txt')
        with open(clean_file_name, 'w') as f:
            f.write(cleaned_metadata)

        processed_samples_count += 1
        processed_samples_list.append(row['sample'])
        print(f"Processed samples count: {processed_samples_count}")
        print("Cleaned metadata:")
        print(cleaned_metadata)
        print("===================================")

    print(f"All processed samples: {processed_samples_list}")
    return metadata_dict





# Sample DataFrame (replace this with your actual dataframe)
samples_df = pd.DataFrame({'sample': [#'SRS969018', #ENVO --> ok
                                      #'DRS005000', # nothing --> ok
                                      #'ERS4036427', # NCBITaxon --> not replaced because non-existent in ENVO.tsv
                                      #'SRS6307425', # NCBITaxon yes, but UBERON not --> because non-existent in ENVO.tsv
                                      #'ERS481159', # PO sample_supplier_name=PO_2732 not translated because non-existent in ENVO.tsv
                                      #'ERS481227', # PO PO_1591 not translated because non-existent in ENVO.tsv
                                      'SRS932724',  # FOODON: one out of two; UBERON: no. Reason: non-existent in ENVO.tsv
                                      'SRS1092766' # FOODON --> ok
                                      ]})  # Modify accordingly

base_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs'  # Replace with your directory path


#label_info_dict = fetch_label_info("https://ontobee.org/listTerms/ENVO?format=tsv")

metadata_dict = process_metadata(samples_df, label_info_dict, base_dir)






