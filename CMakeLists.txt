# cmake global
cmake_minimum_required(VERSION 2.8.12)
# let cmake autolink dependencies on windows
# it's specified globally, cause qt libraries requires that on windows and they are also found globally
cmake_policy(SET CMP0020 NEW)

project(ethereum)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


######################################################################################################

# user defined, defaults
# Normally, set(...CACHE...) creates cache variables, but does not modify them.
function(createDefaultCacheConfig)
	set(HEADLESS OFF CACHE BOOL "Do not compile GUI (AlethZero)")
	set(VMTRACE OFF CACHE BOOL "VM tracing and run-time checks (useful for cross-implementation VM debugging)")
	set(PARANOIA OFF CACHE BOOL "Additional run-time checks")
	set(JSONRPC ON CACHE BOOL "Build with jsonprc. default on")
	set(EVMJIT OFF CACHE BOOL "Build a just-in-time compiler for EVM code (requires LLVM)")
endfunction()


# propagates CMake configuration options to the compiler
function(configureProject)
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

	if (EVMJIT)
		add_definitions(-DETH_EVMJIT)
	endif()

	if (HEADLESS)
		add_definitions(-DETH_HEADLESS)
	endif()
endfunction()



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

	if (EVMJIT)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/JIT")
	else ()
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/int")
	endif ()

	if (PARANOIA)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/PARA")
	endif ()

	#cmake build type may be not specified when using msvc
	if (CMAKE_BUILD_TYPE)
		set(_cmake_build_type ${CMAKE_BUILD_TYPE})
	else()
		set(_cmake_build_type "${CMAKE_CFG_INTDIR}")
	endif()

	# Generate header file containing useful build information
	add_custom_target(BuildInfo.h ALL
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMAND ${CMAKE_COMMAND} -DETH_SOURCE_DIR="${CMAKE_SOURCE_DIR}" -DETH_DST_DIR="${CMAKE_BINARY_DIR}"
			-DETH_BUILD_TYPE="${_cmake_build_type}" -DETH_BUILD_PLATFORM="${ETH_BUILD_PLATFORM}" 
			-P "${ETH_SCRIPTS_DIR}/buildinfo.cmake"
		)
	include_directories(${CMAKE_CURRENT_BINARY_DIR})

	set(CMAKE_INCLUDE_CURRENT_DIR ON)
	set(SRC_LIST BuildInfo.h)
endfunction()



######################################################################################################


set(CMAKE_AUTOMOC ON)
cmake_policy(SET CMP0015 NEW)


createDefaultCacheConfig()
configureProject()
message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")
message("-- VMTRACE: ${VMTRACE}; PARANOIA: ${PARANOIA}; HEADLESS: ${HEADLESS}; JSONRPC: ${JSONRPC}; EVMJIT: ${EVMJIT}")


# Default TARGET_PLATFORM to "linux".
set(TARGET_PLATFORM CACHE STRING "linux")
if ("x${TARGET_PLATFORM}" STREQUAL "x")
	set(TARGET_PLATFORM "linux")
endif ()

if ("${TARGET_PLATFORM}" STREQUAL "linux")
	set(CMAKE_THREAD_LIBS_INIT pthread)
endif ()

include(EthCompilerSettings)
message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")


# this must be an include, as a function it would messs up with variable scope!
include(EthDependencies)
include(EthExecutableHelper)

createBuildInfo()

if (EVMJIT)
	set(EVMJIT_CPP TRUE) # include CPP-JIT connector
	add_subdirectory(evmjit)
endif()

add_subdirectory(libdevcore)
add_subdirectory(libevmcore)
add_subdirectory(liblll)

if (NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
	add_subdirectory(libserpent)
	add_subdirectory(sc)
endif ()

add_subdirectory(libsolidity)
add_subdirectory(lllc)
add_subdirectory(solc)

if (JSONRPC)
	add_subdirectory(libweb3jsonrpc)
endif()

add_subdirectory(secp256k1)
add_subdirectory(libp2p)
add_subdirectory(libdevcrypto)
add_subdirectory(libwhisper)

add_subdirectory(libethcore)
add_subdirectory(libevm)
add_subdirectory(libethereum)

add_subdirectory(libwebthree)
add_subdirectory(test)
add_subdirectory(eth)

if("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
	add_subdirectory(exp)
endif ()

# TODO check msvc
if(NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
	add_subdirectory(neth)
endif ()

if (NOT HEADLESS)
	
	add_subdirectory(libnatspec)
	add_subdirectory(libjsqrc)
	add_subdirectory(libqwebthree)
	add_subdirectory(alethzero)
	add_subdirectory(third)
	add_subdirectory(mix)
	
endif()

enable_testing()
add_test(NAME alltests WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test COMMAND testeth)

#unset(TARGET_PLATFORM CACHE)

if (WIN32)
	# packaging stuff
	include(InstallRequiredSystemLibraries)
	set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ethereum")
	set(CPACK_PACKAGE_VENDOR "ethereum.org")
	set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
	set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
	set(CPACK_PACKAGE_VERSION "0.7")
	set(CPACK_GENERATOR "NSIS")
	# seems to be not working
	# set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.bmp")

	# our stuff
	set(CPACK_COMPONENT_ALETHZERO_GROUP "Applications")
	set(CPACK_COMPONENT_THIRD_GROUP "Applications")
	set(CPACK_COMPONENT_MIX_GROUP "Applications")
	set(CPACK_COMPONENTS_ALL alethzero third mix)

	# nsis specific stuff
	set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} ethereum")
	set(CPACK_NSIS_HELP_LINK "https://github.com/ethereum/cpp-ethereum")
	set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/ethereum/cpp-ethereum")
	set(CPACK_NSIS_CONTACT "ethereum.org")
	set(CPACK_NSIS_MODIFY_PATH ON)
	set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.ico")
	set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.ico")

	include(CPack)
endif (WIN32)
