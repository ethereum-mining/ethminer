FROM ubuntu:14.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -qy wget
RUN apt-get install -qy build-essential g++-4.8 automake libtool unzip git cmake
RUN apt-get install -qy libncurses5-dev libgmp-dev libgmp3-dev libboost-all-dev libleveldb-dev yasm libminiupnpc-dev
RUN apt-get install -qy qtbase5-dev qt5-default qtdeclarative5-dev libqt5webkit5-dev

RUN mkdir /cryptopp562
RUN cd /cryptopp562 && wget http://www.cryptopp.com/cryptopp562.zip && unzip cryptopp562.zip
RUN cd /cryptopp562 && make -j $(cat /proc/cpuinfo | grep processor | wc -l) && make install

RUN git clone --depth=1 https://github.com/ethereum/cpp-ethereum
RUN mkdir cpp-ethereum/build
RUN cd cpp-ethereum/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j $(cat /proc/cpuinfo | grep processor | wc -l) && make install

ENTRYPOINT ["/usr/local/bin/eth"]
