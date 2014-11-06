cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

# dependencies will be installed into this directory
set(ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install")

file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR})

include(cryptopp.cmake)
include(curl.cmake)
include(json-rpc-cpp.cmake)
include(miniupnpc.cmake)

