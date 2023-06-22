#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:10:34 2023

@author: dgaio
"""

import os
import re
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt

def find_pmids(directory):
    pmid_dict = {}
    pmid_pattern = re.compile(r"PMID[^0-9]*\d{8}", re.IGNORECASE)

    file_counter = 0
    start_time = time.time()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    if lines:
                        key = lines[0].strip().replace(">","")
                        pmid_dict[key] = pmid_dict.get(key, [])
                        for line in lines[1:]:
                            match = pmid_pattern.search(line)
                            if match:
                                pmid_dict[key].append(match.group())
                
                file_counter += 1
                
                if file_counter % 1000 == 0:  # print progress every 1000 files
                    elapsed_time = time.time() - start_time
                    print(f"Processed {file_counter} files in {elapsed_time:.2f} seconds "
                          f"({file_counter / elapsed_time:.2f} files/second)")

    return pmid_dict


directory = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
pmid_dict = find_pmids(directory)
# 6000 files per second --> expected: 4h for 2M samples 


def dict_to_csv(dictionary, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample', 'pmid', 'pmid_digits'])

        for key, values in dictionary.items():
            if values:
                pmid = '&'.join(values)
                pmid_digits = re.findall(r'\d+', pmid)  # Extract digits
                first_digits = pmid_digits[0] if pmid_digits else ''  # Get the first digit
                writer.writerow([key, pmid, first_digits])
            else:
                writer.writerow([key, '', ''])



# Call the function to convert the dictionary to CSV
dict_to_csv(pmid_dict, '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid.csv')


# visualize how mahy samples contain pmid info: import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid.csv')


# Count the non-NA values in the 'pmid_digits' column
non_na_count = data['pmid_digits'].notna().sum()

# Count the NA values in the 'pmid_digits' column
na_count = data['pmid_digits'].isna().sum()

# Create a bar plot to visualize the counts
labels = ['Non-NA', 'NA']
counts = [non_na_count, na_count]

plt.bar(labels, counts)
plt.xlabel('Column "pmid_digits"')
plt.ylabel('Count')
plt.title('Count of Non-NA Values in Column "pmid_digits"')

# Display the count on top of each bar
for i, count in enumerate(counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()
