cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

# dependencies will be installed into this directory
set(ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install")


include(cryptopp.cmake)
#include(miniupnpc.cmake)

