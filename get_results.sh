#!/bin/bash

# Nome do contêiner que você quer buscar
CONTAINER_NAME="fl-hiaac_docker-server"

# Diretório dentro do contêiner que você quer copiar
CONTAINER_DIR="results/"

# Diretório de destino na máquina local
LOCAL_DIR="results/"

# Verifica se todos os argumentos foram passados
if [ -z "$CONTAINER_NAME" ] || [ -z "$CONTAINER_DIR" ] || [ -z "$LOCAL_DIR" ]; then
    echo "Uso: $0 <nome_do_container> <diretorio_no_container> <diretorio_local>"
    exit 1
fi

# Obtém o ID do contêiner com base no nome
CONTAINER_ID=$(docker ps -a -q -f "ancestor=$CONTAINER_NAME")

# Verifica se o contêiner foi encontrado
if [ -z "$CONTAINER_ID" ]; then
    echo "Erro: Contêiner com nome '$CONTAINER_NAME' não encontrado."
    exit 1
fi

echo "ID do contêiner encontrado: $CONTAINER_ID"

# Copia a pasta do contêiner para a máquina local
docker cp "$CONTAINER_ID:$CONTAINER_DIR" "./"

sudo chmod -R u+rw "/results"
sudo chown -R gustavo:gustavo results/

echo "aqui" $LOCAL_DIR

# Verifica se a operação de cópia foi bem-sucedida
if [ $? -eq 0 ]; then
    echo "Pasta copiada com sucesso de '$CONTAINER_DIR' para '$LOCAL_DIR'"
else
    echo "Erro ao copiar a pasta."
fi
