# Dockerfile for cpp-ethereum

### Quick usage

    docker run -d ethereum/client-cpp

### Building

Dockerfile to build a cpp-ethereum docker image from source

    docker build -t cpp-ethereum .

### Running

    docker run -d cpp-ethereum

### Usage

First enter the container:

    docker exec -it <container name> bash

Inspect logs:

    cat /var/log/cpp-ethereum.log
    cat /var/log/cpp-ethereum.err

Restart supervisor service:

    supervisorctl restart cpp-ethereum
