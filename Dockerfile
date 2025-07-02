# 1. Use Miniconda3 as the base image
FROM continuumio/miniconda3

# 2. Set working directory in the container
WORKDIR /app

# 3. Copy only what's needed for environment first (for better Docker layer caching)
COPY metadmin_env.yml /app/

# 4. Create the conda environment
RUN conda env create -f /app/metadmin_env.yml

# 5. Copy the rest of the project files (scripts, data, etc.)
COPY . /app

# 6. Activate the conda environment for all subsequent RUN and CMD commands
SHELL ["conda", "run", "-n", "metadmin_env", "/bin/bash", "-c"]

# 7. Install additional Python dependencies (e.g., OpenAI client)
RUN pip install --upgrade pip && \
    pip install openai==1.93.0

# 8. Default command (interactive shell)
CMD ["bash"]

