#!/bin/sh

# Miner account
export ETH_WALLET=<replace with your address>
export WORKER_NAME=<replace with your worker name>

# Start mining!
docker run --gpus all -e ETH_WALLET -e WORKER_NAME -P -it ethminer:latest
