# all dependencies that are not directly included in the cpp-ethereum distribution are defined here
# for this to work, download the dependency via the cmake script in extdep or install them manually!

# by defining this variable, cmake will look for dependencies first in our own repository before looking in system paths like /usr/local/ ...
# this must be set to point to the same directory as $ETH_DEPENDENCY_INSTALL_DIR in /extdep directory
string(TOLOWER ${CMAKE_SYSTEM_NAME} _system_name)
set (CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/extdep/install/${_system_name}")


# Dependencies must have a version number, to ensure reproducible build. The version provided here is the one that is in the extdep repository. If you use system libraries, version numbers may be different.

find_package (CryptoPP 5.6.2 EXACT REQUIRED)
message(" - CryptoPP header: ${CRYPTOPP_INCLUDE_DIRS}")
message(" - CryptoPP lib   : ${CRYPTOPP_LIBRARIES}")

# TODO the Jsoncpp package does not yet check for correct version number
find_package (Jsoncpp 0.60 REQUIRED)
message(" - Jsoncpp header: ${JSONCPP_INCLUDE_DIRS}")
message(" - Jsoncpp lib   : ${JSONCPP_LIBRARIES}")

# TODO the JsonRpcCpp package does not yet check for correct version number
find_package (JsonRpcCpp 0.3.2 REQUIRED)
if (${JSON_RPC_CPP_FOUND})
    message (" - json-rpc-cpp header: ${JSON_RPC_CPP_INCLUDE_DIRS}")
    message (" - json-rpc-cpp lib   : ${JSON_RPC_CPP_LIBRARIES}")
	add_definitions(-DETH_JSONRPC)
endif()
