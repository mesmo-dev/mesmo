# syntax=docker/dockerfile:1

FROM condaforge/miniforge3:latest

WORKDIR /mesmo

# Setup conda environment.
RUN conda create -n mesmo -c conda-forge python=3.8 contextily cvxpy numpy pandas scipy
# Activate conda environment for RUN commands.
SHELL ["conda", "run", "--no-capture-output", "-n", "mesmo", "/bin/bash", "-c"]
# Activate conda environment for interactive shell.
RUN echo "source activate mesmo" > ~/.bashrc

# Install MESMO.
COPY . .
RUN python development_setup.py

# Cleanup caches to reduce image size.
RUN conda clean --all
RUN pip cache purge
