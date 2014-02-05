project(ethereum)
cmake_minimum_required(VERSION 2.8)

cmake_policy(SET CMP0015 NEW)

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Compiler-specific C++11 activation.
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	execute_process(
		COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
	if (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
		message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
	endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
else ()
	message(FATAL_ERROR "Your C++ compiler does not support C++11.")
endif ()

# Look for available Crypto++ version and if it is >= 5.6.2
find_path(ID cryptlib.h
	/usr/include/cryptopp
	/usr/include/crypto++
	/usr/local/include/cryptopp
	/usr/local/include/crypto++
	/opt/local/include/cryptopp
	/opt/local/include/crypto++
	)
find_library(LS NAMES cryptoppeth cryptopp
	PATHS
	/usr/lib
	/usr/local/lib
	/opt/local/lib
	)

if(ID AND LS)
	set(CRYPTOPP_FOUND TRUE)
	message(STATUS "Found Crypto++: ${ID}, ${LS}")
	set(_CRYPTOPP_VERSION_HEADER ${ID}/config.h)
	if(EXISTS ${_CRYPTOPP_VERSION_HEADER})
		file(STRINGS ${_CRYPTOPP_VERSION_HEADER} _CRYPTOPP_VERSION REGEX "^#define CRYPTOPP_VERSION[ \t]+[0-9]+$")
		string(REGEX REPLACE "^#define CRYPTOPP_VERSION[ \t]+([0-9]+)" "\\1" _CRYPTOPP_VERSION ${_CRYPTOPP_VERSION})
		if(${_CRYPTOPP_VERSION} LESS 562)
			message(STATUS "System Crypto++ version found is smaller than 5.6.2.")
		else()
			set(CRYPTOPP_INCLUDE_DIR ${ID} CACHE FILEPATH "" FORCE)
			set(CRYPTOPP_LIBRARIES ${LS} CACHE FILEPATH "" FORCE)
			set(CRYPTOPP_FOUND TRUE)
			message(STATUS "System Crypto++ found and version greater or equal to 5.6.2")
		endif()
	endif()
endif()

if(NOT CRYPTOPP_FOUND)
	message(STATUS "System Crypto++ not found, broken or too old. We use ../cryptopp562")
	set(CRYPTOPP_INCLUDE_DIR "../cryptopp562" CACHE FILEPATH "" FORCE)
	set(CRYPTOPP_LIBRARIES "../cryptopp562" CACHE FILEPATH "" FORCE)
endif()

# Not really worth caching. We want to reevaluate anyway.
mark_as_advanced(CRYPTOPP_INCLUDE_DIR CRYPTOPP_LIBRARIES)

# Always "found", given last block.
include_directories(${CRYPTOPP_INCLUDE_DIR})
link_directories(${CRYPTOPP_LIBRARIES})

add_subdirectory(secp256k1)
add_subdirectory(libethereum)
add_subdirectory(test)
add_subdirectory(eth)
