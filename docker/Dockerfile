FROM ubuntu:14.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get upgrade -y

# Ethereum dependencies
RUN apt-get install -qy build-essential g++-4.8 git cmake libboost-all-dev libcurl4-openssl-dev wget
RUN apt-get install -qy automake unzip libgmp-dev libtool libleveldb-dev yasm libminiupnpc-dev libreadline-dev scons
RUN apt-get install -qy libjsoncpp-dev libargtable2-dev
RUN apt-get install -qy libncurses5-dev libcurl4-openssl-dev wget
RUN apt-get install -qy libjsoncpp-dev libargtable2-dev libmicrohttpd-dev

# Ethereum PPA
RUN apt-get install -qy software-properties-common
RUN add-apt-repository ppa:ethereum/ethereum
RUN add-apt-repository ppa:ethereum/ethereum-dev
RUN apt-get update
RUN apt-get install -qy libcryptopp-dev libjson-rpc-cpp-dev

# LLVM-3.5
RUN wget -O - http://llvm.org/apt/llvm-snapshot.gpg.key|sudo apt-key add -
RUN echo "deb http://llvm.org/apt/trusty/ llvm-toolchain-trusty-3.5 main\ndeb-src http://llvm.org/apt/trusty/ llvm-toolchain-trusty-3.5 main" > /etc/apt/sources.list.d/llvm-trusty.list
RUN apt-get update
RUN apt-get install -qy llvm-3.5 libedit-dev

# Fix llvm-3.5 cmake paths
RUN mkdir -p /usr/lib/llvm-3.5/share/llvm && ln -s /usr/share/llvm-3.5/cmake /usr/lib/llvm-3.5/share/llvm/cmake

# Build Ethereum (HEADLESS)
RUN git clone --depth=1 https://github.com/ethereum/cpp-ethereum
RUN mkdir -p cpp-ethereum/build
RUN cd cpp-ethereum/build && cmake .. -DHEADLESS=1 -DLLVM_DIR=/usr/share/llvm-3.5/cmake -DEVMJIT=1 && make -j $(cat /proc/cpuinfo | grep processor | wc -l) && make install
RUN ldconfig

ENTRYPOINT ["/usr/local/bin/eth"]
