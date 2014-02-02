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

# Override the cryptopp search until we can get the qmake stuff sorted.
set(CRYPTOPP_INCLUDE_DIR "../cryptopp562" CACHE FILEPATH "" FORCE)
set(CRYPTOPP_LIBRARIES "../cryptopp562" CACHE FILEPATH "" FORCE)

# Look for availabe Crypto++ version and if it is >= 5.6.2
if(CRYPTOPP_INCLUDE_DIR AND CRYPTOPP_LIBRARIES)
	   set(CRYPTOPP_FOUND TRUE)
	   message(STATUS "Found Crypto++: ${CRYPTOPP_INCLUDE_DIR}, ${CRYPTOPP_LIBRARIES}")
else()
       find_path(CRYPTOPP_INCLUDE_DIR cryptlib.h
               /usr/include/crypto++
               /usr/include/cryptopp
               /usr/local/include/crypto++
               /usr/local/include/cryptopp
               /opt/local/include/crypto++
               /opt/local/include/cryptopp
               )
       find_library(CRYPTOPP_LIBRARIES NAMES cryptopp
               PATHS
               /usr/lib
               /usr/local/lib
               /opt/local/lib
               )
       if(CRYPTOPP_INCLUDE_DIR AND CRYPTOPP_LIBRARIES)
               set(CRYPTOPP_FOUND TRUE)
               message(STATUS "Found Crypto++: ${CRYPTOPP_INCLUDE_DIR}, ${CRYPTOPP_LIBRARIES}")
               set(_CRYPTOPP_VERSION_HEADER ${CRYPTOPP_INCLUDE_DIR}/config.h)
               if(EXISTS ${_CRYPTOPP_VERSION_HEADER})
                       file(STRINGS ${_CRYPTOPP_VERSION_HEADER} _CRYPTOPP_VERSION REGEX "^#define CRYPTOPP_VERSION[ \t]+[0-9]+$")
                       string(REGEX REPLACE "^#define CRYPTOPP_VERSION[ \t]+([0-9]+)" "\\1" _CRYPTOPP_VERSION ${_CRYPTOPP_VERSION})
                       if(${_CRYPTOPP_VERSION} LESS 562)
                               message(STATUS "Crypto++ version found is smaller than 5.6.2. We use ../cryptopp562")
                               set(CRYPTOPP_INCLUDE_DIR "../cryptopp562" CACHE FILEPATH "" FORCE)
                               set(CRYPTOPP_LIBRARIES "../cryptopp562" CACHE FILEPATH "" FORCE)
                       else()
                               message(STATUS "Crypto++ found and version greater or equal to 5.6.2")
                       endif()
               endif()
       else()
               set(CRYPTOPP_FOUND TRUE) # Ugly
               message(STATUS "Crypto++ not found. We use ../cryptopp562")
			   set(CRYPTOPP_INCLUDE_DIR "../cryptopp562" CACHE FILEPATH "" FORCE)
			   set(CRYPTOPP_LIBRARIES "../cryptopp562" CACHE FILEPATH "" FORCE)
        endif()
        mark_as_advanced(CRYPTOPP_INCLUDE_DIR CRYPTOPP_LIBRARIES)
endif()

if(CRYPTOPP_FOUND)
	include_directories(${CRYPTOPP_INCLUDE_DIR})
	link_directories(${CRYPTOPP_LIBRARIES})
endif()

add_subdirectory(libethereum)
add_subdirectory(test)
add_subdirectory(eth)
