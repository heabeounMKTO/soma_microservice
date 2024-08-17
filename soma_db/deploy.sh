#! /usr/bin/bash


source ./.env

export POSTGRES_PORT=${POSTGRES_PORT}
export POSTGRES_USER=${POSTGRES_USER}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
export POSTGRES_DB=${POSTGRES_DB}
export POSTGRES_DIR=${POSTGRES_DIR}

mkdir -p ${POSTGRES_DIR} 

docker stack deploy --with-registry-auth \
  --resolve-image=always \
  --compose-file docker-compose.yml soma 
