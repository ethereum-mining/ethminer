cmake_minimum_required(VERSION 3.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	# The default toolchain file configures compilers and build environment.
	# This configuration is also used by hunter to build dependencies.
	# CMake will cache this value, not need to explictly specify CACHE param.
	set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/cmake/toolchain.cmake)
endif()

include(HunterGate)
HunterGate(
	URL "https://github.com/ruslo/hunter/archive/v0.18.44.tar.gz"
	SHA1 "a78f0b377b8e53c038f12fc18b0c02564c4534c8"
)


set(GENOIL_VERSION "1.1.7")
cmake_policy(SET CMP0042 OLD) 	# fix MACOSX_RPATH
cmake_policy(SET CMP0048 NEW) 	# allow VERSION argument in project()

project(ethminer VERSION 0.9.41)

# link_directories interprate relative paths with respect to CMAKE_CURRENT_SOURCE_DIR
cmake_policy(SET CMP0015 NEW)

# let cmake autolink dependencies on windows
# it's specified globally, cause qt libraries requires that on windows and they are also found globally
cmake_policy(SET CMP0020 NEW)

# 3.1 and above
cmake_policy(SET CMP0054 NEW)

######################################################################################################

# note: The value "default" which provides the defaults is just a fake value
# which lets us keep the default values of all build options and is set at
# the beginning of this file.

#defaults:
set(D_CMAKE_BUILD_TYPE "Release")
set(D_ETHASHCL ON)
set(D_ETHASHCUDA OFF)
set(D_JSONRPC ON)
set(D_ETHSTRATUM OFF)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE ${D_CMAKE_BUILD_TYPE})
endif ()

# propagates CMake configuration options to the compiler
function(configureProject)

	if (ETHASHCL)
		add_definitions(-DETH_ETHASHCL)
	endif()

	if (ETHASHCUDA)
		add_definitions(-DETH_ETHASHCUDA)
	endif()

	if (ETHSTRATUM)
		add_definitions(-DETH_STRATUM)
	endif()
endfunction()

set(CPPETHEREUM 1)

function(createBuildInfo)
	# Set build platform; to be written to BuildInfo.h
	set(ETH_BUILD_PLATFORM "${TARGET_PLATFORM}")
	if (CMAKE_COMPILER_IS_MINGW)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/mingw")
	elseif (CMAKE_COMPILER_IS_MSYS)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/msys")
	elseif (CMAKE_COMPILER_IS_GNUCXX)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/g++")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/msvc")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/clang")
	else ()
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/unknown")
	endif ()

	set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/int")

	#cmake build type may be not specified when using msvc
	if (CMAKE_BUILD_TYPE)
		set(_cmake_build_type ${CMAKE_BUILD_TYPE})
	else()
		set(_cmake_build_type "${CMAKE_CFG_INTDIR}")
	endif()

	message("createBuildInfo()")

	# Generate header file containing useful build information
	add_custom_target(BuildInfo.h ALL
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMAND ${CMAKE_COMMAND} -DETH_SOURCE_DIR="${CMAKE_SOURCE_DIR}" -DETH_DST_DIR="${CMAKE_BINARY_DIR}"
			-DETH_BUILD_TYPE="${_cmake_build_type}" -DETH_BUILD_PLATFORM="${ETH_BUILD_PLATFORM}"
			-DPROJECT_VERSION="${PROJECT_VERSION}" -DGENOIL_VERSION="${GENOIL_VERSION}"
			-P "${ETH_SCRIPTS_DIR}/buildinfo.cmake"
		)
	include_directories(${CMAKE_CURRENT_BINARY_DIR})

	set(CMAKE_INCLUDE_CURRENT_DIR ON)
	set(SRC_LIST BuildInfo.h)
endfunction()



######################################################################################################

macro(eth_format_option O)
	if (${${O}})
		set(${O} ON)
	elseif ("${${O}}" STREQUAL "" AND ${D_${O}})
		set(${O} ON)
	else()
		set(${O} OFF)
	endif()
endmacro()

# Normalise build options
eth_format_option(JSONRPC)
eth_format_option(ETHASHCL)
eth_format_option(ETHASHCUDA)
eth_format_option(ETHSTRATUM)

# Default CMAKE_BUILD_TYPE accordingly.
set(CMAKE_BUILD_TYPE CACHE STRING ${D_CMAKE_BUILD_TYPE})

# Default TARGET_PLATFORM to ${CMAKE_SYSTEM_NAME}
# change this once we support cross compiling
set(TARGET_PLATFORM CACHE STRING ${CMAKE_SYSTEM_NAME})
if ("x${TARGET_PLATFORM}" STREQUAL "x")
	set(TARGET_PLATFORM ${CMAKE_SYSTEM_NAME})
endif ()

hunter_add_package(Boost COMPONENTS system)
find_package(Boost CONFIG REQUIRED COMPONENTS system)

hunter_add_package(jsoncpp)
find_package(jsoncpp CONFIG REQUIRED)

include(ProjectJsonRpcCpp)

include(EthDependencies)

configureProject()

message("------------------------------------------------------------------------")
message("--                  CMake Version                            ${CMAKE_VERSION}")
message("-- CMAKE_BUILD_TYPE Build type                               ${CMAKE_BUILD_TYPE}")
message("-- TARGET_PLATFORM  Target platform                          ${TARGET_PLATFORM}")
message("--------------------------------------------------------------- features")
message("--                  Chromium support                         ${ETH_HAVE_WEBENGINE}")
message("--                  HTTP Request support                     ${CURL_FOUND}")
message("-- JSONRPC          JSON-RPC support                         ${JSONRPC}")
message("------------------------------------------------------------- components")
message("-- ETHASHCL         Build OpenCL components                  ${ETHASHCL}")
message("-- ETHASHCUDA       Build CUDA components                    ${ETHASHCUDA}")
message("-- ETHSTRATUM       Build Stratum components                 ${ETHSTRATUM}")
message("------------------------------------------------------------------------")
message("")

if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
set(CMAKE_THREAD_LIBS_INIT pthread)
endif()

include(EthCompilerSettings)
message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")

# this must be an include, as a function it would mess up with variable scope!
include(EthExecutableHelper)

message("creating build info...")
createBuildInfo()

add_subdirectory(libdevcore)

add_subdirectory(libethash)
if (ETHASHCL)
	add_subdirectory(libethash-cl)
endif ()
if (ETHASHCUDA)
	add_subdirectory(libethash-cuda)
endif ()
if(ETHSTRATUM)
	add_subdirectory(libstratum)
endif()

add_subdirectory(libethcore)
add_subdirectory(ethminer)
