#!/bin/bash

ETH_DEPENDENCY_SOURCE_DIR=$1
ETH_DEPENDENCY_INSTALL_DIR=$2

OLD_COMMON_DYLIB="libjsonrpccpp-common.0.dylib"

COMMON_DYLIB=${ETH_DEPENDENCY_INSTALL_DIR}/lib/libjsonrpccpp-common.0.dylib
SERVER_DYLIB=${ETH_DEPENDENCY_INSTALL_DIR}/lib/libjsonrpccpp-server.0.dylib
CLIENT_DYLIB=${ETH_DEPENDENCY_INSTALL_DIR}/lib/libjsonrpccpp-client.0.dylib

# fix bin
STAB_EXEC=${ETH_DEPENDENCY_INSTALL_DIR}/bin/jsonrpcstub 
install_name_tool -change ${OLD_COMMON_DYLIB} ${COMMON_DYLIB} ${STAB_EXEC}

# fix common
install_name_tool -id ${COMMON_DYLIB} ${COMMON_DYLIB}

# fix server
install_name_tool -id ${SERVER_DYLIB} ${SERVER_DYLIB}
install_name_tool -change ${OLD_COMMON_DYLIB} ${COMMON_DYLIB} ${SERVER_DYLIB}

# fix client
install_name_tool -id ${CLIENT_DYLIB} ${CLIENT_DYLIB}
install_name_tool -change ${OLD_COMMON_DYLIB} ${COMMON_DYLIB} ${CLIENT_DYLIB}

# TODO fix argtable and jsoncpp once they are downloaded as dependencies


