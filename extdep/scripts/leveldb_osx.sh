#!/bin/bash

ETH_DEPENDENCY_SOURCE_DIR=$1
ETH_DEPENDENCY_INSTALL_DIR=$2

OLD_SNAPPY_DYLIB="/Users/marekkotewicz/ethereum/cpp-ethereum/extdep/install/darwin/lib/libsnappy.1.dylib"
SNAPPY_DYLIB=${ETH_DEPENDENCY_INSTALL_DIR}/lib/libsnappy.dylib 
LEVELDB_DYLIB=${ETH_DEPENDENCY_INSTALL_DIR}/lib/libleveldb.dylib

install_name_tool -id ${LEVELDB_DYLIB} ${LEVELDB_DYLIB}
install_name_tool -change ${OLD_SNAPPY_DYLIB} ${SNAPPY_DYLIB} ${LEVELDB_DYLIB}

