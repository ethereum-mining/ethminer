# cmake global
cmake_minimum_required(VERSION 2.8.12)

project(ethereum)

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

# user defined, defaults
# Normally, set(...CACHE...) creates cache variables, but does not modify them.
option(VMTRACE "VM tracing and run-time checks (useful for cross-implementation VM debugging)" OFF)
option(PARANOID "Additional run-time checks" OFF)
option(JSONRPC "Build with jsonprc. default on" ON)
option(FATDB "Build with ability to list entries in the Trie. Doubles DB size, slows everything down, but good for looking at state diffs and trie contents." OFF)
option(USENPM "Use npm to recompile ethereum.js if it was changed" OFF)
option(PROFILING "Build in support for profiling" OFF)
option(ROCKSDB "Use rocksdb rather than leveldb" OFF)

set(BUNDLE "none" CACHE STRING "Predefined bundle of software to build (none, full, user, tests, minimal).")
option(MINER "Build the CLI miner component" ON)
option(ETHKEY "Build the CLI key manager component" ON)
option(SOLIDITY "Build the Solidity language components" ON)
option(SERPENT "Build the Serpent language components" ON)
option(TOOLS "Build the tools components" ON)
option(NCURSES "Build the NCurses components" OFF)
option(GUI "Build GUI components (AlethZero, Mix)" ON)
option(TESTS "Build the tests." ON)
option(NOBOOST "No use of boost macros in test functions" OFF)
option(EVMJIT "Build just-in-time compiler for EVM code (requires LLVM)" OFF)
option(ETHASHCL "Build in support for GPU mining via OpenCL" OFF)
option(JSCONSOLE "Build in javascript console" OFF)

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

	if (CPUID_FOUND)
		add_definitions(-DETH_CPUID)
	endif()

	if (CURL_FOUND)
		add_definitions(-DETH_CURL)
	endif()

	if (NOBOOST)
		add_definitions(-DNOBOOST)
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

	if (EVMJIT)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/JIT")
	else ()
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/int")
	endif ()

	if (PARANOID)
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

# TODO: Abstract into something sensible and move into a function.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(DECENT_PLATFORM OFF)
else ()
	set(DECENT_PLATFORM ON)
endif ()

macro(eth_format_option O)
	if (${${O}})
		set(${O} ON)	
	else()
		set(${O} OFF)	
	endif()
endmacro()

macro(eth_format_option_on_decent_platform O)
	if (${${O}})
		set(${O} ${DECENT_PLATFORM})
	else()
		set(${O} OFF)
	endif()
endmacro()

# Normalise build options
eth_format_option(PARANOID)
eth_format_option(VMTRACE)
eth_format_option(EVMJIT)
eth_format_option(FATDB)
eth_format_option(JSONRPC)
eth_format_option(MINER)
eth_format_option(USENPM)
eth_format_option(PROFILING)
eth_format_option(SOLIDITY)
eth_format_option(ROCKSDB)
eth_format_option(GUI)
eth_format_option(TESTS)
eth_format_option(NOBOOST)
eth_format_option(ROCKSDB)
eth_format_option(TOOLS)
eth_format_option(ETHKEY)
eth_format_option(ETHASHCL)
eth_format_option(JSCONSOLE)
eth_format_option_on_decent_platform(SERPENT)
eth_format_option_on_decent_platform(NCURSES)

if (JSCONSOLE)
	set(JSONRPC ON)
endif()

if (GUI)
	set(JSONRPC ON)
endif()

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
#	set(NCURSES ${DECENT_PLATFORM})
	set(TOOLS ON)
	set(TESTS ON)
	set(FATDB ON)
elseif (BUNDLE STREQUAL "cli")
	set(SERPENT ${DECENT_PLATFORM})
	set(SOLIDITY ON)
	set(USENPM ON)
	set(GUI OFF)
#	set(NCURSES ${DECENT_PLATFORM})
	set(TOOLS ON)
	set(TESTS ON)
	set(FATDB ON)
elseif (BUNDLE STREQUAL "core")
	set(SERPENT OFF)
	set(SOLIDITY ON)
	set(USENPM OFF)
	set(GUI ON)
	set(NCURSES OFF)
	set(TOOLS ON)
	set(TESTS OFF)
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
#	set(NCURSES ${DECENT_PLATFORM})
	set(TOOLS ON)
	set(TESTS OFF)
elseif (BUNDLE STREQUAL "wallet")
	set(SERPENT OFF)
	set(SOLIDITY OFF)
	set(USENPM OFF)
	set(GUI OFF)
	set(NCURSES OFF)
	set(TOOLS OFF)
	set(TESTS OFF)
	set(ETHKEY ON)
	set(MINER OFF)
	set(ETHASHCL ON)
elseif (BUNDLE STREQUAL "miner")
	set(SERPENT OFF)
	set(SOLIDITY OFF)
	set(USENPM OFF)
	set(GUI OFF)
	set(NCURSES OFF)
	set(TOOLS OFF)
	set(TESTS OFF)
	set(ETHKEY OFF)
	set(MINER ON)
	set(ETHASHCL ON)
endif ()

# Default CMAKE_BUILD_TYPE to "Release".
set(CMAKE_BUILD_TYPE CACHE STRING "Release")
if ("x${CMAKE_BUILD_TYPE}" STREQUAL "x")
	set(CMAKE_BUILD_TYPE "Release")
endif ()

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
message("-- NOBOOST          No BOOST macros in test functions        ${NOBOOST}")
message("-- FATDB            Full database exploring                  ${FATDB}")
message("-- JSONRPC          JSON-RPC support                         ${JSONRPC}")
message("-- USENPM           Javascript source building               ${USENPM}")
message("-- ROCKSDB          Prefer rocksdb to leveldb                ${ROCKSDB}")
message("------------------------------------------------------------- components")
message("-- MINER            Build miner                              ${MINER}")
message("-- ETHKEY           Build wallet tools                       ${ETHKEY}")
message("-- TOOLS            Build basic tools                        ${TOOLS}")
message("-- SOLIDITY         Build Solidity language components       ${SOLIDITY}")
message("-- SERPENT          Build Serpent language components        ${SERPENT}")
message("-- GUI              Build GUI components                     ${GUI}")
message("-- NCURSES          Build NCurses components                 ${NCURSES}")
message("-- TESTS            Build tests                              ${TESTS}")
message("-- ETHASHCL         Build OpenCL components (experimental!)  ${ETHASHCL}")
message("-- JSCONSOLE        Build with javascript console            ${JSCONSOLE}")
message("-- EVMJIT           Build LLVM-based JIT EVM (experimental!) ${EVMJIT}")
message("------------------------------------------------------------------------")
message("")

set(CMAKE_THREAD_LIBS_INIT pthread)

include(EthCompilerSettings)
message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")

# this must be an include, as a function it would mess up with variable scope!
include(EthExecutableHelper)

createBuildInfo()

if (ROCKSDB AND ROCKSDB_FOUND)
	set(DB_INCLUDE_DIRS ${ROCKSDB_INCLUDE_DIRS})
	set(DB_LIBRARIES ${ROCKSDB_LIBRARIES})
	add_definitions(-DETH_ROCKSDB)
else()
	set(DB_INCLUDE_DIRS ${LEVELDB_INCLUDE_DIRS})
	set(DB_LIBRARIES ${LEVELDB_LIBRARIES})
endif()

if (EVMJIT)
	set(EVMJIT_CPP TRUE) # include CPP-JIT connector
	add_subdirectory(evmjit)
endif()

if (TOOLS OR GUI OR SOLIDITY OR NCURSES OR TESTS)
	set(GENERAL 1)
else ()
	set(GENERAL 0)
endif ()

message("GENERAL ${GENERAL}")

add_subdirectory(libdevcore)
if (GENERAL)
	add_subdirectory(libevmcore)
	add_subdirectory(libevmasm)
	add_subdirectory(liblll)
endif ()

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
endif ()

if (JSONRPC AND GENERAL)
	add_subdirectory(libweb3jsonrpc)
endif ()

if (JSCONSOLE)
	add_subdirectory(libjsengine)
	add_subdirectory(libjsconsole)
	add_subdirectory(ethconsole)
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
endif ()

add_subdirectory(libethcore)

if (GENERAL)
	add_subdirectory(libevm)
	add_subdirectory(libethereum)
	add_subdirectory(libwebthree)
endif ()

if (MINER OR TOOLS)
	add_subdirectory(ethminer)
endif ()

if (ETHKEY OR TOOLS)
	add_subdirectory(ethkey)
endif ()

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
	add_subdirectory(ethvm)
	add_subdirectory(eth)

	if("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
		add_subdirectory(exp)
	endif ()

endif()

#if (NCURSES)
#	add_subdirectory(neth)
#endif ()

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

if (APPLE AND GUI)

	add_custom_target(appdmg
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMAND ${CMAKE_COMMAND}
		-DAPP_DMG_EXE=${ETH_APP_DMG}
		-DAPP_DMG_FILE=appdmg.json.in
		-DAPP_DMG_ICON="alethzero/alethzero.icns"
		-DAPP_DMG_BACKGROUND="install-folder-bg.png"
		-DETH_BUILD_DIR="${CMAKE_BINARY_DIR}"
		-DETH_MIX_APP="$<TARGET_FILE_DIR:mix>"
		-DETH_ALETHZERO_APP="$<TARGET_FILE_DIR:AlethZero>"
		-P "${ETH_SCRIPTS_DIR}/appdmg.cmake"
	)

endif ()

if (WIN32)
	# packaging stuff
	include(InstallRequiredSystemLibraries)
	set(CPACK_PACKAGE_NAME "Ethereum")
	set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Ethereum")
	set(CPACK_PACKAGE_VENDOR "ethereum.org")
	set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
	set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
	set(CPACK_PACKAGE_VERSION "0.9")
	set(CPACK_GENERATOR "NSIS")
	# seems to be not working
	# set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.bmp")

	# our stuff
	set(CPACK_COMPONENT_ALETHZERO_GROUP "Applications")
	set(CPACK_COMPONENT_MIX_GROUP "Applications")
	set(CPACK_COMPONENT_SOLC_GROUP "CLI")
	set(CPACK_COMPONENT_ETH_GROUP "CLI")
	set(CPACK_COMPONENT_ETHMINER_GROUP "CLI")
	set(CPACK_COMPONENT_RLP_GROUP "CLI")
	set(CPACK_COMPONENT_ABI_GROUP "CLI")

	set(CPACK_COMPONENTS_ALL alethzero mix solc eth ethminer rlp abi)

	# nsis specific stuff
	if (CMAKE_CL_64)
		set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
		set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} (Win64)")
	else ()
		set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
		set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
	endif()

	set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} Ethereum")
	set(CPACK_NSIS_HELP_LINK "https://github.com/ethereum/cpp-ethereum")
	set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/ethereum/cpp-ethereum")
	set(CPACK_NSIS_CONTACT "ethereum.org")
	set(CPACK_NSIS_MODIFY_PATH ON)
	set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.ico")
	set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.ico")

	include(CPack)
endif (WIN32)
