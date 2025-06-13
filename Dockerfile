# Use Miniconda as base
FROM continuumio/miniconda3

# Set working directory in the container
WORKDIR /app

# Copy project files into container
COPY . /app

# Create Conda environment
RUN conda env create -f metadmin_env.yml

# Activate environment by default
SHELL ["conda", "run", "-n", "metadmin_env", "/bin/bash", "-c"]

# Set the entry point (shell by default, can be changed per container)
CMD ["bash"]

