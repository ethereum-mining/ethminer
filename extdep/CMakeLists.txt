cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

# all dependencies will be installed into this directory, separated by platform
string(TOLOWER ${CMAKE_SYSTEM_NAME} _system_name)
set(ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install/${_system_name}")
file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/lib)
file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/bin)

include(icu.cmake)
include(jom.cmake)
include(qt.cmake)
include(curl.cmake)
include(jsoncpp.cmake)
include(argtable2.cmake)
include(json-rpc-cpp.cmake)
include(cryptopp.cmake)
include(snappy.cmake)
include(leveldb.cmake)

# will be re-eanbled later
# include(miniupnpc.cmake)


# BUG
# becouse extdep/cmakelist.txt is subfolder of ethereum project its being install into binary destination on macos. it's a bug

