# Building Ethereum

## Dependencies

secp256k1 implementation: https://github.com/sipa/secp256k1.git
Expects secp256k1 directory to be in same path as cpp-ethereum.

(NOTE: secp256k1 requires a development installation of the GMP library, libssl and libcrypto++.)

libcrypto++, version 5.6.2 or greater (i.e. with SHA3 support). Because it's so recent, it expects this to be built in a directory libcrypto562 in the same path as cpp-ethereum. A recent version of Boost (I use version 1.53) and leveldb (I use version 1.9.0).

A decent C++11 compiler (I use GNU GCC 4.8.1). CMake, version 2.8 or greater.

On Ubuntu:

	sudo apt-get install libgmp3-dev libcrypto++-dev libssl-dev libboost-all-dev cmake libleveldb-dev  libminiupnpc-dev

## Building

	mkdir /path/to/cpp-ethereum/../cpp-ethereum-build
	cd /path/to/cpp-ethereum-build
	cmake -DCMAKE_BUILD_TYPE=Debug /path/to/cpp-ethereum
	make


