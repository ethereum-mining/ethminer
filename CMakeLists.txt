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
	set(VMTRACE OFF CACHE BOOL "VM tracing and run-time checks (useful for cross-implementation VM debugging)")
	set(PARANOIA OFF CACHE BOOL "Additional run-time checks")
	set(JSONRPC ON CACHE BOOL "Build with jsonprc. default on")
	set(FATDB OFF CACHE BOOL "Build with ability to list entries in the Trie. Doubles DB size, slows everything down, but good for looking at state diffs and trie contents.")
	set(USENPM OFF CACHE BOOL "Use npm to recompile ethereum.js if it was changed")
	set(PROFILING OFF CACHE BOOL "Build in support for profiling")

	set(BUNDLE "none" CACHE STRING "Predefined bundle of software to build (none, full, user, tests, minimal).")
	set(SOLIDITY ON CACHE BOOL "Build the Solidity language components")
	set(SERPENT ON CACHE BOOL "Build the Serpent language components")
	set(TOOLS ON CACHE BOOL "Build the tools components")
	set(NCURSES ON CACHE BOOL "Build the NCurses components")
	set(GUI ON CACHE BOOL "Build GUI components (AlethZero, Mix)")
	set(TESTS ON CACHE BOOL "Build the tests.")
	set(EVMJIT OFF CACHE BOOL "Build just-in-time compiler for EVM code (requires LLVM)")
	set(ETHASHCL OFF CACHE BOOL "Build in support for GPU mining via OpenCL")
endfunction()


# propagates CMake configuration options to the compiler
function(configureProject)
	if (PARANOIA)
		add_definitions(-DETH_PARANOIA)
	endif ()

	if (VMTRACE)
		add_definitions(-DETH_VMTRACE)
	endif ()

	if (ETHASHCL)
		add_definitions(-DETH_ETHASHCL)
	endif()

	if (EVMJIT)
		add_definitions(-DETH_EVMJIT)
	endif()

	if (FATDB)
		add_definitions(-DETH_FATDB)
	endif()

	if (SOLIDITY)
		add_definitions(-DETH_SOLIDITY)
	endif()

	if (GUI)
		add_definitions(-DETH_GUI)
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

# Clear invalid option
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	if (PARANOIA)
		message("Paranoia requires debug - disabling for release build.")
		set(PARANOIA OFF)
	endif ()
	if (VMTRACE)
		message("VM Tracing requires debug - disabling for release build.")
		set (VMTRACE OFF)
	endif ()
endif ()

createDefaultCacheConfig()
configureProject()

# Force chromium.
set (ETH_HAVE_WEBENGINE 1)

# Normalise build options
# TODO: Abstract into something sensible and move into a function.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(DECENT_PLATFORM OFF)
else ()
	set(DECENT_PLATFORM ON)
endif ()
# Backwards compatibility
if (HEADLESS)
	set(BUNDLE "minimal")
endif ()

if (PARANOIA)
	set(PARANOIA ON)
else ()
	set(PARANOIA OFF)
endif ()
if (VMTRACE)
	set(VMTRACE ON)
else ()
	set(VMTRACE OFF)
endif ()
if (EVMJIT)
	set(EVMJIT ON)
else ()
	set(EVMJIT OFF)
endif()
if (FATDB)
	set(FATDB ON)
else ()
	set(FATDB OFF)
endif()
if (JSONRPC)
	set(JSONRPC ON)
else ()
	set(JSONRPC OFF)
endif ()
if (USENPM)
	set(USENPM ON)
else ()
	set(USENPM OFF)
endif ()
if (PROFILING)
	set(PROFILING ON)
else ()
	set(PROFILING OFF)
endif ()


if (SOLIDITY)
	set(SOLIDITY ON)
else ()
	set(SOLIDITY OFF)
endif()
if (SERPENT)
	set(SERPENT ${DECENT_PLATFORM})
else ()
	set(SERPENT OFF)
endif()
if (GUI)
	set(GUI ON)
	set(JSONRPC ON)
else ()
	set(GUI OFF)
endif ()
if (TESTS)
	set(TESTS ON)
else ()
	set(TESTS OFF)
endif ()
if (TOOLS)
	set(TOOLS ON)
else ()
	set(TOOLS OFF)
endif ()
if (ETHASHCL)
	set(ETHASHCL ON)
else ()
	set(ETHASHCL OFF)
endif()
if (NCURSES)
	set(NCURSES ${DECENT_PLATFORM})
else ()
	set(NCURSES OFF)
endif ()

if (BUNDLE STREQUAL "minimal")
	set(SERPENT OFF)
	set(SOLIDITY OFF)
	set(USENPM OFF)
	set(GUI OFF)
	set(NCURSES OFF)
	set(TOOLS ON)
	set(TESTS OFF)
elseif (BUNDLE STREQUAL "full")
	set(SERPENT ${DECENT_PLATFORM})
	set(SOLIDITY ON)
	set(USENPM ON)
	set(GUI ON)
	set(NCURSES ${DECENT_PLATFORM})
	set(TOOLS ON)
	set(TESTS ON)
	set(FATDB ON)
elseif (BUNDLE STREQUAL "tests")
	set(SERPENT ${DECENT_PLATFORM})
	set(SOLIDITY ON)
	set(USENPM OFF)
	set(GUI OFF)
	set(NCURSES OFF)
	set(TOOLS OFF)
	set(TESTS ON)
	set(FATDB ON)
elseif (BUNDLE STREQUAL "user")
	set(SERPENT OFF)
	set(SOLIDITY OFF)
	set(USENPM OFF)
	set(GUI ON)
	set(NCURSES ${DECENT_PLATFORM})
	set(TOOLS ON)
	set(TESTS OFF)
endif ()

# Default CMAKE_BUILD_TYPE to "Release".
set(CMAKE_BUILD_TYPE CACHE STRING "Release")
if ("x${CMAKE_BUILD_TYPE}" STREQUAL "x")
	set(CMAKE_BUILD_TYPE "Release")
endif ()

# Default TARGET_PLATFORM to "linux".
set(TARGET_PLATFORM CACHE STRING "linux")
if ("x${TARGET_PLATFORM}" STREQUAL "x")
	set(TARGET_PLATFORM "linux")
endif ()

message("------------------------------------------------------------------------")
message("--                  CMake Version                            ${CMAKE_VERSION}")
message("-- CMAKE_BUILD_TYPE Build type                               ${CMAKE_BUILD_TYPE}")
message("-- TARGET_PLATFORM  Target platform                          ${TARGET_PLATFORM}")
message("-- BUNDLE           Build bundle                             ${BUNDLE}")
message("--------------------------------------------------------------- features")
message("--                  Chromium support                         ${ETH_HAVE_WEBENGINE}")
message("-- VMTRACE          VM execution tracing                     ${VMTRACE}")
message("-- PROFILING        Profiling support                        ${PROFILING}")
message("-- PARANOIA         Additional (SLOW) database checking      ${PARANOIA}")
message("-- FATDB            Full database exploring                  ${FATDB}")
message("-- JSONRPC          JSON-RPC support                         ${JSONRPC}")
message("-- USENPM           Javascript source building               ${USENPM}")
message("------------------------------------------------------------- components")
message("-- TOOLS            Build basic tools                        ${TOOLS}")
message("-- SOLIDITY         Build Solidity language components       ${SOLIDITY}")
message("-- SERPENT          Build Serpent language components        ${SERPENT}")
message("-- GUI              Build GUI components                     ${GUI}")
message("-- NCURSES          Build NCurses components                 ${NCURSES}")
message("-- TESTS            Build tests                              ${TESTS}")
message("-- ETHASHCL         Build OpenCL components (experimental!)  ${ETHASHCL}")
message("-- EVMJIT           Build LLVM-based JIT EVM (experimental!) ${EVMJIT}")
message("------------------------------------------------------------------------")
message("")


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

if (SERPENT)
	add_subdirectory(libserpent)
	add_subdirectory(sc)
endif ()

if (SOLIDITY)
	add_subdirectory(libsolidity)
endif ()

if (TOOLS)
	add_subdirectory(lllc)
	if (SOLIDITY)
		add_subdirectory(solc)
	endif ()
endif()

if (JSONRPC)
	add_subdirectory(libweb3jsonrpc)
endif()

add_subdirectory(secp256k1)
add_subdirectory(libp2p)
add_subdirectory(libdevcrypto)
add_subdirectory(libwhisper)

add_subdirectory(libethash)
if (ETHASHCL)
	add_subdirectory(libethash-cl)
endif ()

add_subdirectory(libethcore)
add_subdirectory(libevm)
add_subdirectory(libethereum)
add_subdirectory(libwebthree)

if (TESTS)
	add_subdirectory(libtestutils)
	add_subdirectory(test)
	if (JSONRPC)
		add_subdirectory(ethrpctest)
	endif ()
endif ()

if (TOOLS)

	add_subdirectory(rlp)
	add_subdirectory(abi)
	add_subdirectory(eth)

	if("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
		add_subdirectory(exp)
	endif ()

endif()

if (NCURSES)
	add_subdirectory(neth)
endif ()

if (GUI)

	add_subdirectory(libnatspec)
	add_subdirectory(libjsqrc)

	if (ETH_HAVE_WEBENGINE)
		add_subdirectory(alethzero)
#		add_subdirectory(third)	// reenable once not qtwebkit.
	endif()

	if (SOLIDITY)
		add_subdirectory(mix)
	endif ()

endif()

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
