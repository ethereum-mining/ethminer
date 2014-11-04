FROM ubuntu:14.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get upgrade -y

# Ethereum dependencies
RUN apt-get install -qy build-essential g++-4.8 git cmake libboost-all-dev libcurl4-openssl-dev wget
RUN apt-get install -qy automake unzip libgmp-dev libtool libleveldb-dev yasm libminiupnpc-dev libreadline-dev scons

# NCurses based GUI (not optional though for a succesful compilation, see https://github.com/ethereum/cpp-ethereum/issues/452 )
RUN apt-get install -qy libncurses5-dev

# Qt-based GUI
# RUN apt-get install -qy qtbase5-dev qt5-default qtdeclarative5-dev libqt5webkit5-dev

# Cryptopp
RUN git clone --depth=1 https://github.com/mmoss/cryptopp.git
RUN cd cryptopp && scons --shared --prefix=/usr

# JSONRPC (version 0.2.1, see https://github.com/ethereum/cpp-ethereum/issues/453 )
RUN apt-get install -qy libjsoncpp-dev libargtable2-dev
RUN git clone --depth=1 --branch=0.2.1 https://github.com/cinemast/libjson-rpc-cpp.git
RUN mkdir -p libjson-rpc-cpp/build
RUN cd libjson-rpc-cpp/build && cmake .. && make && make install

# Build Ethereum (HEADLESS)
RUN git clone --depth=1 https://github.com/ethereum/cpp-ethereum
RUN mkdir -p cpp-ethereum/build
RUN cd cpp-ethereum/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DHEADLESS=1 && make -j $(cat /proc/cpuinfo | grep processor | wc -l) && make install
RUN ldconfig

ENTRYPOINT ["/usr/local/bin/eth"]
