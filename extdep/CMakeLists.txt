cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

# all dependencies will be installed into this directory, separated by platform
string(TOLOWER ${CMAKE_SYSTEM_NAME} _system_name)
set(ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install/${_system_name}")
file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR})

include(cryptopp.cmake)
# will be re-eanbled later
# include(curl.cmake)
include(json-rpc-cpp.cmake)
#include(miniupnpc.cmake)
