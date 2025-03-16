FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Define o comando padrão para executar nvidia-smi
CMD ["nvidia-smi"]

RUN apt-get update && apt-get install -y \
    python3-pip \
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

# Ativar o ambiente virtual e instalar as dependências do requirements.txt
RUN pip install --no-cache-dir  -r /app/requirements.txt
