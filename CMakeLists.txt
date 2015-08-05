# cmake global
cmake_minimum_required(VERSION 2.8.12)

set(PROJECT_VERSION "0.9.39")
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

# TODO: Abstract into something sensible and move into a function.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(DECENT_PLATFORM OFF)
else ()
	set(DECENT_PLATFORM ON)
endif ()

#defaults:
set(D_CMAKE_BUILD_TYPE "RelWithDebInfo")
set(D_SERPENT ${DECENT_PLATFORM})
set(D_SOLIDITY ON)
set(D_USENPM OFF)
set(D_GUI ON)
set(D_TOOLS ON)
set(D_TESTS ON)
set(D_FATDB ON)
set(D_ETHASHCL ON)
set(D_EVMJIT ON)
set(D_JSCONSOLE ON)
set(D_JSONRPC ON)
set(D_VMTRACE OFF)
set(D_PARANOID OFF)
set(D_PROFILING OFF)
set(D_ROCKSDB OFF)
set(D_NOBOOST OFF)
set(D_FRONTIER ON)
set(D_MINER ON)
set(D_ETHKEY ON)

if (BUNDLE STREQUAL "minimal")
	set(D_SERPENT OFF)
	set(D_SOLIDITY OFF)
	set(D_USENPM OFF)
	set(D_GUI OFF)
	set(D_TOOLS ON)
	set(D_TESTS OFF)
elseif (BUNDLE STREQUAL "full")
	set(D_SERPENT ${DECENT_PLATFORM})
	set(D_SOLIDITY ON)
	set(D_USENPM ON)
	set(D_GUI ON)
	set(D_TOOLS ON)
	set(D_TESTS ON)
	set(D_FATDB ON)
elseif (BUNDLE STREQUAL "cli")
	set(D_SERPENT ${DECENT_PLATFORM})
	set(D_SOLIDITY ON)
	set(D_USENPM ON)
	set(D_GUI OFF)
	set(D_TOOLS ON)
	set(D_TESTS ON)
	set(D_FATDB ON)
elseif (BUNDLE STREQUAL "core")
	set(D_SERPENT OFF)
	set(D_SOLIDITY ON)
	set(D_USENPM OFF)
	set(D_GUI ON)
	set(D_TOOLS ON)
	set(D_TESTS OFF)
	set(D_FATDB ON)
elseif (BUNDLE STREQUAL "tests")
	set(D_SERPENT ${DECENT_PLATFORM})
	set(D_SOLIDITY ON)
	set(D_USENPM OFF)
	set(D_GUI OFF)
	set(D_TOOLS OFF)
	set(D_TESTS ON)
	set(D_FATDB ON)
elseif (BUNDLE STREQUAL "user")
	set(D_SERPENT OFF)
	set(D_SOLIDITY OFF)
	set(D_USENPM OFF)
	set(D_GUI ON)
	set(D_TOOLS ON)
	set(D_TESTS OFF)
elseif (BUNDLE STREQUAL "wallet")
	set(D_SERPENT OFF)
	set(D_SOLIDITY OFF)
	set(D_USENPM OFF)
	set(D_GUI OFF)
	set(D_TOOLS OFF)
	set(D_TESTS OFF)
	set(D_ETHKEY ON)
	set(D_MINER OFF)
	set(D_ETHASHCL ON)
elseif (BUNDLE STREQUAL "miner")
	set(D_SERPENT OFF)
	set(D_SOLIDITY OFF)
	set(D_USENPM OFF)
	set(D_GUI OFF)
	set(D_TOOLS OFF)
	set(D_TESTS OFF)
	set(D_ETHKEY OFF)
	set(D_MINER ON)
	set(D_ETHASHCL ON)
elseif (BUNDLE STREQUAL "release")      # release builds
	set(D_SERPENT ${DECENT_PLATFORM})
	set(D_SOLIDITY ON)
	set(D_USENPM OFF)
	set(D_GUI ON)
	set(D_TOOLS ON)
	set(D_TESTS OFF)
	set(D_FATDB OFF)
	set(D_ETHASHCL ON)
	set(D_EVMJIT ON)
	set(D_JSCONSOLE ON)
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

	if (FRONTIER)
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
			-DPROJECT_VERSION="${PROJECT_VERSION}"
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
eth_format_option(FRONTIER)
eth_format_option(SERPENT)

if (JSCONSOLE)
	set(JSONRPC ON)
endif()

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
message("-- NOBOOST          No BOOST macros in test functions        ${NOBOOST}")
message("-- FATDB            Full database exploring                  ${FATDB}")
message("-- JSONRPC          JSON-RPC support                         ${JSONRPC}")
message("-- USENPM           Javascript source building               ${USENPM}")
message("-- ROCKSDB          Prefer rocksdb to leveldb                ${ROCKSDB}")
message("-- FRONTIER         Default to the Frontier network          ${FRONTIER}")
message("------------------------------------------------------------- components")
message("-- MINER            Build miner                              ${MINER}")
message("-- ETHKEY           Build wallet tools                       ${ETHKEY}")
message("-- TOOLS            Build basic tools                        ${TOOLS}")
message("-- SOLIDITY         Build Solidity language components       ${SOLIDITY}")
message("-- SERPENT          Build Serpent language components        ${SERPENT}")
message("-- GUI              Build GUI components                     ${GUI}")
message("-- TESTS            Build tests                              ${TESTS}")
message("-- ETHASHCL         Build OpenCL components                  ${ETHASHCL}")
message("-- JSCONSOLE        Build with javascript console            ${JSCONSOLE}")
message("-- EVMJIT           Build LLVM-based JIT EVM                 ${EVMJIT}")
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
	if (NOT DEFINED LLVM_DIR)
		if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
			set(LLVM_DIR "${CMAKE_SOURCE_DIR}/extdep/install/windows/x64/share/llvm/cmake")
		elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
			set(LLVM_DIR "/usr/local/opt/llvm/share/llvm/cmake")
		endif()
	endif()

	set(EVMJIT_CPP TRUE) # include CPP-JIT connector
	add_subdirectory(evmjit)
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
		set(EVMJIT_DLLS_LOCAL $<TARGET_FILE:evmjit>)
		set(EVMJIT_DLLS optimized ${EVMJIT_DLLS_LOCAL} debug ${EVMJIT_DLLS_LOCAL})
	endif()
endif()

if (TOOLS OR GUI OR SOLIDITY OR TESTS)
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
	set(CPACK_PACKAGE_NAME "Ethereum (++)")
	set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The Ethereum (++) Toolset")
	set(CPACK_PACKAGE_VENDOR "ethereum.org")
	set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
	set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
	set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
	set(CPACK_GENERATOR "NSIS")
	# seems to be not working
	# set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.bmp")

	# our stuff
	#set(CPACK_COMPONENT_ALETHZERO_GROUP "Applications")
	#set(CPACK_COMPONENT_MIX_GROUP "Applications")
	#set(CPACK_COMPONENT_SOLC_GROUP "CLI")
	#set(CPACK_COMPONENT_ETH_GROUP "CLI")
	#set(CPACK_COMPONENT_ETHMINER_GROUP "CLI")
	#set(CPACK_COMPONENT_RLP_GROUP "CLI")
	#set(CPACK_COMPONENT_ABI_GROUP "CLI")

	#set(CPACK_COMPONENTS_ALL alethzero mix solc eth ethminer rlp abi)

	# nsis specific stuff
	if (CMAKE_CL_64)
		set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
		set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} (Win64)")
	else ()
		set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
		set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
	endif()

	set(CPACK_NSIS_DISPLAY_NAME "Ethereum (++)")
	set(CPACK_NSIS_HELP_LINK "https://github.com/ethereum/cpp-ethereum")
	set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/ethereum/cpp-ethereum")
	set(CPACK_NSIS_CONTACT "ethereum.org")
	set(CPACK_NSIS_MODIFY_PATH ON)
	set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.ico")
	set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/alethzero/alethzero.ico")

	include(CPack)
endif (WIN32)
