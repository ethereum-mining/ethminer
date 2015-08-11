# all dependencies that are not directly included in the cpp-ethereum distribution are defined here
# for this to work, download the dependency via the cmake script in extdep or install them manually!

# by defining this variable, cmake will look for dependencies first in our own repository before looking in system paths like /usr/local/ ...
# this must be set to point to the same directory as $ETH_DEPENDENCY_INSTALL_DIR in /extdep directory
string(TOLOWER ${CMAKE_SYSTEM_NAME} _system_name)
if (CMAKE_CL_64)
	set (ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extdep/install/${_system_name}/x64")
else ()
	set (ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extdep/install/${_system_name}/Win32")
endif()
set (CMAKE_PREFIX_PATH ${ETH_DEPENDENCY_INSTALL_DIR})

# setup directory for cmake generated files and include it globally 
# it's not used yet, but if we have more generated files, consider moving them to ETH_GENERATED_DIR
set(ETH_GENERATED_DIR "${PROJECT_BINARY_DIR}/gen")
include_directories(${ETH_GENERATED_DIR})

# custom cmake scripts
set(ETH_SCRIPTS_DIR ${CMAKE_SOURCE_DIR}/cmake/scripts)

# Qt5 requires opengl
# TODO use proper version of windows SDK (32 vs 64)
# TODO make it possible to use older versions of windows SDK (7.0+ should also work)
# TODO it windows SDK is NOT FOUND, throw ERROR
# from https://github.com/rpavlik/cmake-modules/blob/master/FindWindowsSDK.cmake
if (WIN32)
	find_package(WINDOWSSDK REQUIRED)
	message(" - WindowsSDK dirs: ${WINDOWSSDK_DIRS}")
	set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${WINDOWSSDK_DIRS})
endif()

# homebrew installs qts in opt
if (APPLE)
	set (CMAKE_PREFIX_PATH "/usr/local/opt/qt5" ${CMAKE_PREFIX_PATH})
	set (CMAKE_PREFIX_PATH "/usr/local/opt/v8-315" ${CMAKE_PREFIX_PATH})
endif()

find_program(CTEST_COMMAND ctest)
message(STATUS "ctest path: ${CTEST_COMMAND}")

# Dependencies must have a version number, to ensure reproducible build. The version provided here is the one that is in the extdep repository. If you use system libraries, version numbers may be different.

find_package (CryptoPP 5.6.2 EXACT REQUIRED)
message(" - CryptoPP header: ${CRYPTOPP_INCLUDE_DIRS}")
message(" - CryptoPP lib   : ${CRYPTOPP_LIBRARIES}")

find_package (LevelDB REQUIRED)
message(" - LevelDB header: ${LEVELDB_INCLUDE_DIRS}")
message(" - LevelDB lib: ${LEVELDB_LIBRARIES}")

find_package (RocksDB)
if (ROCKSDB_FOUND)
	message(" - RocksDB header: ${ROCKSDB_INCLUDE_DIRS}")
	message(" - RocksDB lib: ${ROCKSDB_LIBRARIES}")
endif()

if (JSCONSOLE)
	find_package (v8 REQUIRED)
	message(" - v8 header: ${V8_INCLUDE_DIRS}")
	message(" - v8 lib   : ${V8_LIBRARIES}")
	add_definitions(-DETH_JSCONSOLE)
endif()

# TODO the Jsoncpp package does not yet check for correct version number
find_package (Jsoncpp 0.60 REQUIRED)
message(" - Jsoncpp header: ${JSONCPP_INCLUDE_DIRS}")
message(" - Jsoncpp lib   : ${JSONCPP_LIBRARIES}")

# TODO get rid of -DETH_JSONRPC
# TODO add EXACT once we commit ourselves to cmake 3.x
if (JSONRPC)
	find_package (json_rpc_cpp 0.4 REQUIRED)
	message (" - json-rpc-cpp header: ${JSON_RPC_CPP_INCLUDE_DIRS}")
	message (" - json-rpc-cpp lib   : ${JSON_RPC_CPP_LIBRARIES}")
	add_definitions(-DETH_JSONRPC)

 	find_package(MHD) 
	message(" - microhttpd header: ${MHD_INCLUDE_DIRS}")
	message(" - microhttpd lib   : ${MHD_LIBRARIES}")
	message(" - microhttpd dll   : ${MHD_DLLS}")
endif() #JSONRPC

# TODO readline package does not yet check for correct version number
# TODO make readline package dependent on cmake options
# TODO get rid of -DETH_READLINE
find_package (Readline 6.3.8)
if (READLINE_FOUND)
	message (" - readline header: ${READLINE_INCLUDE_DIRS}")
	message (" - readline lib   : ${READLINE_LIBRARIES}")
	add_definitions(-DETH_READLINE)
endif ()

# TODO miniupnpc package does not yet check for correct version number
# TODO make miniupnpc package dependent on cmake options
# TODO get rid of -DMINIUPNPC
find_package (Miniupnpc 1.8.2013)
if (MINIUPNPC_FOUND)
	message (" - miniupnpc header: ${MINIUPNPC_INCLUDE_DIRS}")
	message (" - miniupnpc lib   : ${MINIUPNPC_LIBRARIES}")
	add_definitions(-DETH_MINIUPNPC)
endif()

# TODO gmp package does not yet check for correct version number
# TODO it is also not required in msvc build
find_package (Gmp 6.0.0)
if (GMP_FOUND)
	message(" - gmp header: ${GMP_INCLUDE_DIRS}")
	message(" - gmp lib   : ${GMP_LIBRARIES}")
endif()

# curl is only requried for tests
# TODO specify min curl version, on windows we are currently using 7.29
find_package (CURL)
message(" - curl header: ${CURL_INCLUDE_DIRS}")
message(" - curl lib   : ${CURL_LIBRARIES}")

# cpuid required for eth
find_package (Cpuid)
if (CPUID_FOUND)
	message(" - cpuid header: ${CPUID_INCLUDE_DIRS}")
	message(" - cpuid lib   : ${CPUID_LIBRARIES}")
endif()

find_package (OpenCL)
if (OpenCL_FOUND)
	message(" - opencl header: ${OpenCL_INCLUDE_DIRS}")
	message(" - opencl lib   : ${OpenCL_LIBRARIES}")
endif()

# find location of jsonrpcstub
find_program(ETH_JSON_RPC_STUB jsonrpcstub)
message(" - jsonrpcstub location    : ${ETH_JSON_RPC_STUB}")

# do not compile GUI
if (GUI)

# we need json rpc to build alethzero
	if (NOT JSON_RPC_CPP_FOUND)
		message (FATAL_ERROR "JSONRPC is required for GUI client")
	endif()

# find all of the Qt packages
# remember to use 'Qt' instead of 'QT', cause unix is case sensitive
# TODO make headless client optional

	set (ETH_QT_VERSION 5.4)

	find_package (Qt5Core ${ETH_QT_VERSION} REQUIRED)
	find_package (Qt5Gui ${ETH_QT_VERSION} REQUIRED)
	find_package (Qt5Quick ${ETH_QT_VERSION} REQUIRED)
	find_package (Qt5Qml ${ETH_QT_VERSION} REQUIRED)
	find_package (Qt5Network ${ETH_QT_VERSION} REQUIRED)
	find_package (Qt5Widgets ${ETH_QT_VERSION} REQUIRED)
	find_package (Qt5WebEngine ${ETH_QT_VERSION} REQUIRED)
	find_package (Qt5WebEngineWidgets ${ETH_QT_VERSION} REQUIRED)

	# we need to find path to macdeployqt on mac
	if (APPLE)
		set (MACDEPLOYQT_APP ${Qt5Core_DIR}/../../../bin/macdeployqt)
		message(" - macdeployqt path: ${MACDEPLOYQT_APP}")
	endif()
	# we need to find path to windeployqt on windows
	if (WIN32)
		set (WINDEPLOYQT_APP ${Qt5Core_DIR}/../../../bin/windeployqt)
		message(" - windeployqt path: ${WINDEPLOYQT_APP}")
	endif()

	if (APPLE)
		find_program(ETH_APP_DMG appdmg)
		message(" - appdmg location : ${ETH_APP_DMG}")
	endif()

	if (USENPM)

		# TODO check node && npm version
		find_program(ETH_NODE node)
		string(REGEX REPLACE "node" "" ETH_NODE_DIRECTORY ${ETH_NODE})
		message(" - nodejs location : ${ETH_NODE}")

		find_program(ETH_NPM npm)
		string(REGEX REPLACE "npm" "" ETH_NPM_DIRECTORY ${ETH_NPM})
		message(" - npm location    : ${ETH_NPM}")

		if (NOT ETH_NODE)
			message(FATAL_ERROR "node not found!")
		endif()
		if (NOT ETH_NPM)
			message(FATAL_ERROR "npm not found!")
		endif()
	endif()

endif() #GUI

# use multithreaded boost libraries, with -mt suffix
set(Boost_USE_MULTITHREADED ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

# TODO hanlde other msvc versions or it will fail find them
	set(Boost_COMPILER -vc120)
# use static boost libraries *.lib
	set(Boost_USE_STATIC_LIBS ON) 

elseif (APPLE)

# use static boost libraries *.a
	set(Boost_USE_STATIC_LIBS ON) 

elseif (UNIX)
# use dynamic boost libraries .dll
	set(Boost_USE_STATIC_LIBS OFF) 

endif()

find_package(Boost 1.54.0 REQUIRED COMPONENTS thread date_time system regex chrono filesystem unit_test_framework program_options random)

message(" - boost header: ${Boost_INCLUDE_DIRS}")
message(" - boost lib   : ${Boost_LIBRARIES}")

if (APPLE)
	link_directories(/usr/local/lib)
	include_directories(/usr/local/include)
endif()

