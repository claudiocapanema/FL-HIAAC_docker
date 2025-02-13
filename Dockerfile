# Use an official Python runtime as a parent image
#FROM python:3.11-slim-buster

FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Instalar dependências
#RUN apt-get update && apt-get install -y \
#    python3-pip \
#    python3-dev \
#    python3-setuptools \
#    && rm -rf /var/lib/apt/lists/*



# Define o comando padrão para executar nvidia-smi
CMD ["nvidia-smi"]

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install gcc and other dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*
#RUN apt-get update && apt-get install -y nvidia-utils-550
#RUN python3 -m venv /app/venv

# Ativar o ambiente virtual e instalar as dependências do requirements.txt
RUN pip install --no-cache-dir  -r /app/requirements.txt

# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Definir o comando para rodar o script Python (com o ambiente virtual ativado)
#CMD ["bash", "-c", "source /app/venv/bin/activate && python3 app/main.py"]
