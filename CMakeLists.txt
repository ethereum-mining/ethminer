# cmake global
cmake_minimum_required(VERSION 2.8.9)

project(ethereum)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


######################################################################################################

# user defined, defaults
# Normally, set(...CACHE...) creates cache variables, but does not modify them.
function(createDefaultCacheConfig)
	set(HEADLESS OFF CACHE BOOL "Do not compile GUI (AlethZero)")
	set(LANGUAGES OFF CACHE BOOL "Limit build to Serpent/LLL tools")
	set(VMTRACE OFF CACHE BOOL "VM tracing and run-time checks (useful for cross-implementation VM debugging)")
	set(PARANOIA OFF CACHE BOOL "Additional run-time checks")
endfunction()


# propagates CMake configuration options to the compiler
function(configureProject)
	if (LANGUAGES)
		add_definitions(-DETH_LANGUAGES)
	endif ()

	if (PARANOIA)
		if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
			add_definitions(-DETH_PARANOIA)
		else ()
			message(FATAL_ERROR "Paranoia requires debug.")
		endif ()
	endif ()

	if (VMTRACE)
		if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
			add_definitions(-DETH_VMTRACE)
		else ()
			message(FATAL_ERROR "VM tracing requires debug.")
		endif ()
	endif ()
endfunction()


function(createBuildInfo)
	# Set build platform; to be written to BuildInfo.h
	if (CMAKE_COMPILER_IS_MINGW)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/mingw")
	elseif (CMAKE_COMPILER_IS_MSYS)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/msys")
	elseif (CMAKE_COMPILER_IS_GNUCXX)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/g++")
	elseif (CMAKE_COMPILER_IS_MSVC)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/msvc")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/clang")
	else ()
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/unknown")
	endif ()

	# Generate header file containing useful build information
	add_custom_target(BuildInfo.h ALL COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/BuildInfo.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BUILD_TYPE} ${ETH_BUILD_PLATFORM})
	include_directories(${CMAKE_CURRENT_BINARY_DIR})

	set(CMAKE_INCLUDE_CURRENT_DIR ON)
	set(SRC_LIST BuildInfo.h)
endfunction()

######################################################################################################


set(CMAKE_AUTOMOC ON)
cmake_policy(SET CMP0015 NEW)


createDefaultCacheConfig()
configureProject()
message("-- LANGUAGES: ${LANGUAGES}; VMTRACE: ${VMTRACE}; PARANOIA: ${PARANOIA}; HEADLESS: ${HEADLESS}")


# Default TARGET_PLATFORM to "linux".
set(TARGET_PLATFORM CACHE STRING "linux")
if ("x${TARGET_PLATFORM}" STREQUAL "x")
	set(TARGET_PLATFORM "linux")
endif ()

if ("${TARGET_PLATFORM}" STREQUAL "linux")
	set(CMAKE_THREAD_LIBS_INIT pthread)
endif ()

# Set default build type to Release w/debug info
# if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
#    set(CMAKE_BUILD_TYPE RelWithDebInfo)
# endif()


include(EthCompilerSettings)
message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")

#add_definitions("-DETH_BUILD_TYPE=${ETH_BUILD_TYPE}")
#add_definitions("-DETH_BUILD_PLATFORM=${ETH_BUILD_PLATFORM}")


include(EthDependenciesDeprecated)

createBuildInfo()

add_subdirectory(libdevcore)
add_subdirectory(libevmface)
add_subdirectory(liblll)
add_subdirectory(libserpent)
add_subdirectory(libsolidity)
if(NOT APPLE)
	if(PYTHON_LS)
		add_subdirectory(libpyserpent)
	endif()
endif()

add_subdirectory(lllc)
add_subdirectory(solc)
add_subdirectory(sc)
if (NOT LANGUAGES)
	add_subdirectory(secp256k1)
	add_subdirectory(libp2p)
	add_subdirectory(libdevcrypto)
	add_subdirectory(libwhisper)

	add_subdirectory(libethcore)
	add_subdirectory(libevm)
	add_subdirectory(libethereum)
#	add_subdirectory(libethereumx)	 # TODO remove

	add_subdirectory(libwebthree)
	add_subdirectory(test)
	add_subdirectory(eth)
	if("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
		add_subdirectory(exp)
	endif ()
	if(NOT ("${TARGET_PLATFORM}" STREQUAL "w64"))
		add_subdirectory(neth)
	endif ()
	if(QTQML)
		add_definitions(-DETH_QTQML)
	endif()

	if(NOT HEADLESS)
		if ("${TARGET_PLATFORM}" STREQUAL "w64")
			cmake_policy(SET CMP0020 NEW)
		endif ()
		
		add_subdirectory(libqethereum)
		add_subdirectory(alethzero)
		add_subdirectory(third)
		if(QTQML)
			#add_subdirectory(iethxi)
			#add_subdirectory(walleth)	// resurect once we want to submit ourselves to QML.
		endif()
	endif()
endif()


enable_testing()
add_test(NAME alltests WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test COMMAND testeth)

#unset(TARGET_PLATFORM CACHE)

