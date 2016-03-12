# cmake global
cmake_minimum_required(VERSION 2.8.12)

set(PROJECT_VERSION "0.9.41")
set(GENOIL_VERSION "1.0.4")
if (${CMAKE_VERSION} VERSION_GREATER 3.0)
	cmake_policy(SET CMP0042 OLD) 	# fix MACOSX_RPATH
	cmake_policy(SET CMP0048 NEW) 	# allow VERSION argument in project()
	project(ethereum VERSION ${PROJECT_VERSION})
else()
	project(ethereum)
endif()

set(CMAKE_AUTOMOC ON)

# link_directories interprate relative paths with respect to CMAKE_CURRENT_SOURCE_DIR
cmake_policy(SET CMP0015 NEW)

# let cmake autolink dependencies on windows
# it's specified globally, cause qt libraries requires that on windows and they are also found globally
cmake_policy(SET CMP0020 NEW)

# 3.1 and above
if ((${CMAKE_MAJOR_VERSION} GREATER 2) AND (${CMAKE_MINOR_VERSION} GREATER 0))
	# implicitly dereference variables (deprecated in 3.1)
	cmake_policy(SET CMP0054 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

######################################################################################################

# note: The value "default" which provides the defaults is just a fake value
# which lets us keep the default values of all build options and is set at
# the beginning of this file.

#defaults:
set(D_CMAKE_BUILD_TYPE "Release")
set(D_GUI ON)
set(D_TOOLS ON)
set(D_TESTS ON)
set(D_FATDB ON)
set(D_ETHASHCL ON)
set(D_ETHASHCUDA OFF)
set(D_JSONRPC ON)
set(D_VMTRACE OFF)
set(D_PARANOID OFF)
set(D_PROFILING OFF)
set(D_ROCKSDB OFF)
set(D_OLYMPIC OFF)
set(D_MINER ON)
set(D_ETHSTRATUM OFF)

if (BUNDLE STREQUAL "minimal")
	set(D_GUI OFF)
	set(D_TOOLS ON)
	set(D_TESTS OFF)
elseif (BUNDLE STREQUAL "full")
	set(D_GUI ON)
	set(D_TOOLS ON)
	set(D_TESTS ON)
	set(D_FATDB ON)
elseif (BUNDLE STREQUAL "cli")
	set(D_GUI OFF)
	set(D_TOOLS ON)
	set(D_TESTS ON)
	set(D_FATDB ON)
elseif (BUNDLE STREQUAL "core")
	set(D_GUI ON)
	set(D_TOOLS ON)
	set(D_TESTS OFF)
	set(D_FATDB ON)
elseif (BUNDLE STREQUAL "tests")
	set(D_GUI OFF)
	set(D_TOOLS OFF)
	set(D_TESTS ON)
	set(D_FATDB ON)
elseif (BUNDLE STREQUAL "user")
	set(D_GUI ON)
	set(D_TOOLS ON)
	set(D_TESTS OFF)
elseif (BUNDLE STREQUAL "wallet")
	set(D_GUI OFF)
	set(D_TOOLS OFF)
	set(D_TESTS OFF)
	set(D_MINER OFF)
	set(D_ETHASHCL OFF)
	set(D_FATDB OFF)
	set(D_JSONRPC OFF)
elseif (BUNDLE STREQUAL "miner")
	set(D_GUI OFF)
	set(D_TOOLS OFF)
	set(D_TESTS OFF)
	set(D_MINER ON)
	set(D_ETHASHCL ON)
	set(D_FATDB OFF)
	set(D_JSONRPC ON)
	set(D_ETHSTRATUM ON)
elseif (BUNDLE STREQUAL "cudaminer")
	set(D_GUI OFF)
	set(D_TOOLS OFF)
	set(D_TESTS OFF)
	set(D_MINER ON)
	set(D_ETHASHCL ON)
	set(D_ETHASHCUDA ON)
	set(D_FATDB OFF)
	set(D_JSONRPC ON)
	set(D_ETHSTRATUM ON)
elseif (BUNDLE STREQUAL "release")      # release builds
	set(D_GUI ON)
	set(D_TOOLS ON)
	set(D_TESTS OFF)
	set(D_FATDB OFF)
	set(D_ETHASHCL ON)
	set(D_JSONRPC ON)
	set(D_CMAKE_BUILD_TYPE "Release")
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE ${D_CMAKE_BUILD_TYPE})
endif ()

# propagates CMake configuration options to the compiler
function(configureProject)
	if (PARANOID)
		add_definitions(-DETH_PARANOIA)
	endif ()

	if (VMTRACE)
		add_definitions(-DETH_VMTRACE)
	endif ()

	if (ETHASHCL)
		add_definitions(-DETH_ETHASHCL)
	endif()

	if (ETHASHCUDA)
		add_definitions(-DETH_ETHASHCUDA)
	endif()

	if (ETHSTRATUM)
		add_definitions(-DETH_STRATUM)
	endif()

	if (FATDB)
		add_definitions(-DETH_FATDB)
	endif()

	if (GUI)
		add_definitions(-DETH_GUI)
	endif()

	if (CPUID_FOUND)
		add_definitions(-DETH_CPUID)
	endif()

	if (CURL_FOUND)
		add_definitions(-DETH_CURL)
	endif()

	if (OLYMPIC)
		add_definitions(-DETH_OLYMPIC)
	else()
		add_definitions(-DETH_FRONTIER)
	endif()

	add_definitions(-DETH_TRUE)
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

	if (PARANOID)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/PARA")
	endif ()

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
			-DPROJECT_VERSION="${PROJECT_VERSION}" -DGENOIL_VERSION="${GENOIL_VERSION}" -DETH_FATDB="${FATDB}"
			-P "${ETH_SCRIPTS_DIR}/buildinfo.cmake"
		)
	include_directories(${CMAKE_CURRENT_BINARY_DIR})

	set(CMAKE_INCLUDE_CURRENT_DIR ON)
	set(SRC_LIST BuildInfo.h)
endfunction()



######################################################################################################


# Clear invalid option
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	if (PARANOID)
		message("Paranoia requires debug - disabling for release build.")
		set(PARANOID OFF)
	endif ()
	if (VMTRACE)
		message("VM Tracing requires debug - disabling for release build.")
		set (VMTRACE OFF)
	endif ()
endif ()

# Force chromium.
set (ETH_HAVE_WEBENGINE 1)

# Backwards compatibility
if (HEADLESS)
	message("*** WARNING: -DHEADLESS=1 option is DEPRECATED! Use -DBUNDLE=minimal or -DGUI=0")
	set(GUI OFF)
endif ()

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
eth_format_option(PARANOID)
eth_format_option(VMTRACE)
eth_format_option(FATDB)
eth_format_option(JSONRPC)
eth_format_option(MINER)
eth_format_option(PROFILING)
eth_format_option(ROCKSDB)
eth_format_option(GUI)
eth_format_option(TESTS)
eth_format_option(ROCKSDB)
eth_format_option(TOOLS)
eth_format_option(ETHASHCL)
eth_format_option(ETHASHCUDA)
eth_format_option(OLYMPIC)
eth_format_option(ETHSTRATUM)

if (GUI)
	set(JSONRPC ON)
endif()

# Default CMAKE_BUILD_TYPE accordingly.
set(CMAKE_BUILD_TYPE CACHE STRING ${D_CMAKE_BUILD_TYPE})

# Default TARGET_PLATFORM to ${CMAKE_SYSTEM_NAME}
# change this once we support cross compiling
set(TARGET_PLATFORM CACHE STRING ${CMAKE_SYSTEM_NAME})
if ("x${TARGET_PLATFORM}" STREQUAL "x")
	set(TARGET_PLATFORM ${CMAKE_SYSTEM_NAME})
endif ()

include(EthDependencies)

configureProject()

message("------------------------------------------------------------------------")
message("--                  CMake Version                            ${CMAKE_VERSION}")
message("-- CMAKE_BUILD_TYPE Build type                               ${CMAKE_BUILD_TYPE}")
message("-- TARGET_PLATFORM  Target platform                          ${TARGET_PLATFORM}")
message("-- BUNDLE           Build bundle                             ${BUNDLE}")
message("--------------------------------------------------------------- features")
message("--                  Chromium support                         ${ETH_HAVE_WEBENGINE}")
message("--                  Hardware identification support          ${CPUID_FOUND}")
message("--                  HTTP Request support                     ${CURL_FOUND}")
message("-- VMTRACE          VM execution tracing                     ${VMTRACE}")
message("-- PROFILING        Profiling support                        ${PROFILING}")
message("-- FATDB            Full database exploring                  ${FATDB}")
message("-- JSONRPC          JSON-RPC support                         ${JSONRPC}")
message("-- ROCKSDB          Prefer rocksdb to leveldb                ${ROCKSDB}")
message("-- OLYMPIC          Default to the Olympic network           ${OLYMPIC}")
message("------------------------------------------------------------- components")
message("-- MINER            Build miner                              ${MINER}")
message("-- TOOLS            Build basic tools                        ${TOOLS}")
message("-- GUI              Build GUI components                     ${GUI}")
message("-- TESTS            Build tests                              ${TESTS}")
message("-- ETHASHCL         Build OpenCL components                  ${ETHASHCL}")
message("-- ETHASHCUDA       Build CUDA components                    ${ETHASHCUDA}")
message("-- ETHSTRATUM       Build Stratum components (experimental)  ${ETHSTRATUM}")
message("------------------------------------------------------------------------")
message("")

set(CMAKE_THREAD_LIBS_INIT pthread)

include(EthCompilerSettings)
message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")

# this must be an include, as a function it would mess up with variable scope!
include(EthExecutableHelper)

message("creating build info...")
createBuildInfo()

if (ROCKSDB AND ROCKSDB_FOUND)
	set(DB_INCLUDE_DIRS ${ROCKSDB_INCLUDE_DIRS})
	set(DB_LIBRARIES ${ROCKSDB_LIBRARIES})
	add_definitions(-DETH_ROCKSDB)
else()
	set(DB_INCLUDE_DIRS ${LEVELDB_INCLUDE_DIRS})
	set(DB_LIBRARIES ${LEVELDB_LIBRARIES})
endif()

if (TOOLS OR GUI OR TESTS)
	set(GENERAL 1)
else ()
	set(GENERAL 0)
endif ()

add_subdirectory(libdevcore)
if (GENERAL)
	add_subdirectory(libevmcore)
	add_subdirectory(libevmasm)
	add_subdirectory(liblll)
endif ()

if (TOOLS)
	add_subdirectory(lllc)
endif ()

if (NOT WIN32)
	add_definitions(-DETH_HAVE_SECP256K1)
	add_subdirectory(secp256k1)
endif ()

add_subdirectory(libscrypt)
add_subdirectory(libdevcrypto)

if (GENERAL)
	add_subdirectory(libp2p)
	add_subdirectory(libwhisper)
endif ()

if (GENERAL OR MINER)
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
endif ()

add_subdirectory(libethcore)

if (GENERAL)
	add_subdirectory(libevm)
	add_subdirectory(libethereum)
endif ()

if (MINER OR TOOLS)
	add_subdirectory(ethminer)
endif ()
