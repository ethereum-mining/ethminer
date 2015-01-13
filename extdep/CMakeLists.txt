cmake_minimum_required(VERSION 2.8.12)

include(ExternalProject)
include(CMakeParseArguments)
include(eth_download.cmake)

# all dependencies will be installed into this directory, separated by platform
string(TOLOWER ${CMAKE_SYSTEM_NAME} _system_name)
set(ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install/${_system_name}")
set(ETH_DEPENDENCY_SERVER "http://build.ethdev.com/builds/${_system_name}-precompiled")
file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/lib)
file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/bin)

if (ETH_COMPILE)
	# json-rpc-cpp and its dependencies
	include(compile/jsoncpp.cmake)
	include(compile/argtable2.cmake)
	include(compile/curl.cmake)
	include(compile/json-rpc-cpp.cmake)

	# qt and its dependencies 
	include(compile/icu.cmake)
	include(compile/jom.cmake)
	include(compile/qt.cmake)

	# leveldb and its dependencies
	include(compile/snappy.cmake)
	include(compile/leveldb.cmake)

	# cryptopp
	include(compile/cryptopp.cmake)

	# boost
	include(compile/boost.cmake)
else()	
	eth_download(jsoncpp)
	eth_download(json-rpc-cpp OSX_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/json-rpc-cpp_osx.sh)
	
	if (APPLE)
		eth_download(snappy OSX_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/snappy_osx.sh)
	endif()

	eth_download(leveldb OSX_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/leveldb_osx.sh)
	eth_download(qt)
	eth_download(cryptopp)
	eth_download(boost)
	eth_download(curl)

endif()

# will be re-eanbled later
# include(miniupnpc.cmake)
# if install phase of extep fails, even if libs are already created, the ethereum install will fail

