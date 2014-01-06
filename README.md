# ethereum

Ethereum C++ Client.

Gav Wood, 2014.

## Dependencies

secp256k1 implementation: https://github.com/sipa/secp256k1.git
Expects secp256k1 directory to be in same path as cpp-ethereum.

(NOTE: secp256k1 requires a development installation of the GMP library, libssl and
apparently libcrypto++.)

A decent C++11 compiler (I use GNU GCC 4.8.1) and a recent version of Boost (I use version 1.53).

CMake, version 2.8 or greater.

On Ubuntu:

	sudo apt-get install libgmp3-dev libcrypto++-dev libssl-dev libboost-all-dev cmake

## Building

mkdir /path/to/cpp-ethereum/../cpp-ethereum-build
cd /path/to/cpp-ethereum-build
cmake -DCMAKE_BUILD_TYPE=Debug /path/to/cpp-ethereum
make

## Contributing

Read CodingStandards.txt thoroughly, before making alterations to the code base.

Do *NOT* use an editor that automatically reformats whitespace away from astylerc or
the formating guidelines as describled in CodingStandards.txt. Your contributions will be
refused and your commits will be reversed.

