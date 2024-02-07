####################
#
# These below useful when testing embeddings_from_metadata.py 
#
####################



# checking how embeddings_from_metadata.py ran: 

import pstats

# Load the stats file
p = pstats.Stats('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/profile_output')

# Sort the data by cumulative time spent in the function and print the top results
p.sort_stats('cumulative').print_stats(10)

# You can also sort by the total time spent in the function
p.sort_stats('time').print_stats(10)




####################

# time taken between first and last generated file in directory: 

import os
import time

# Specify the directory path
directory = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/20240205_dummy_batch100_float32_temp'

# Get a list of files in the directory
files = os.listdir(directory)

# Filter out only regular files (not directories or special files)
files = [f for f in files if os.path.isfile(os.path.join(directory, f))]

if len(files) < 2:
    print("There are not enough files in the directory to calculate the time difference.")
else:
    # Sort the files by creation time (oldest to newest)
    files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))

    # Get the creation timestamp of the first and last files
    first_timestamp = os.path.getctime(os.path.join(directory, files[0]))
    last_timestamp = os.path.getctime(os.path.join(directory, files[-1]))

    # Calculate the time difference in seconds
    time_passed = last_timestamp - first_timestamp

    # Convert seconds to a human-readable format
    time_passed_formatted = time.strftime("%H hours %M minutes %S seconds", time.gmtime(time_passed))

    print("Time passed between the creation of the first file and the last file:", time_passed_formatted)




####################


# checking out embeddings if as expected: 


import pickle
import numpy as np

# Path to one of your .pkl files
pkl_file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/20240202_real_batch100_temp/embeddings_batch_20240202150005895492.pkl'  # Update this to the actual file path

# Load the data from the pickle file
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# Check the content of the loaded data
for sample_id, embedding in data.items():
    
    print(sample_id)
    print(embedding)
    # Verify that the key is a sample ID (adjust the condition based on your sample ID format)
    assert isinstance(sample_id, str), "Sample ID is not a string"
    
    # Verify that the value is a numpy array of the expected length
    assert isinstance(embedding, np.ndarray), f"Embedding for sample {sample_id} is not a numpy array"
    assert embedding.shape == (1536,), f"Embedding for sample {sample_id} has unexpected shape: {embedding.shape}"

print(f"Checked {len(data)} items. All items have the correct format.")


####################


# comparing how similar emebddings (for the same samples) are, when using 100 vs 500 batch size: 

import pickle
import numpy as np
from scipy.spatial.distance import cosine

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_cosine_similarity(vec1, vec2):
    # Using 1 - cosine distance to get cosine similarity
    return 1 - cosine(vec1, vec2)

def compare_embeddings(embeddings1, embeddings2):
    similarities = []
    for key in embeddings1:
        if key in embeddings2:
            sim = compute_cosine_similarity(embeddings1[key], embeddings2[key])
            similarities.append(sim)
    return similarities

# Adjust the file paths according to your setup
file_path_100 = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/20240202_real_batch100_temp/combined_data.pkl"
file_path_500 = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/20240202_real_batch500_temp/combined_data.pkl"

embeddings_100 = load_embeddings(file_path_100)
embeddings_500 = load_embeddings(file_path_500)


similarities = compare_embeddings(embeddings_100, embeddings_500)

# Compute average similarity
average_similarity = np.mean(similarities)

print(f"Average Cosine Similarity: {average_similarity:.4f}")

# Optional: analyze distribution of similarities
import matplotlib.pyplot as plt

plt.hist(similarities, bins=50)
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Similarities between Embeddings')
plt.show()





