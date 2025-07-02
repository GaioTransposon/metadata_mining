# 1. Use Miniconda3 as the base image
FROM continuumio/miniconda3

# 2. Set working directory inside the container
WORKDIR /MicrobeAtlasProject

# 3. Copy only the environment definition first
COPY metadmin_env.yml /app/

# 4. Create the conda environment
RUN conda env create -f /app/metadmin_env.yml

# 5. Copy the rest of the repo (e.g. scripts, README, etc.)
COPY . /app

# 6. Set default shell to use the conda environment
SHELL ["conda", "run", "-n", "metadmin_env", "/bin/bash", "-c"]

# 7. Optional: install external pip packages
RUN pip install --upgrade pip && \
    pip install openai==1.93.0

# 8. Default command (interactive shell)
ENTRYPOINT ["conda", "run", "-n", "metadmin_env"]
CMD ["bash"]
