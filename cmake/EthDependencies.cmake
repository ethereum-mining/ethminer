# all dependencies that are not directly included in the cpp-ethereum distribution are defined here
# for this to work, download the dependency via the cmake script in extdep or install them manually!

# by defining this variable, cmake will look for dependencies first in our own repository before looking in system paths like /usr/local/ ...
# this must be set to point to the same directory as $ETH_DEPENDENCY_INSTALL_DIR in /extdep directory
string(TOLOWER ${CMAKE_SYSTEM_NAME} _system_name)
set (CMAKE_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extdep/install/${_system_name}")
set (CMAKE_PREFIX_PATH ${CMAKE_DEPENDENCY_INSTALL_DIR})

# Qt5 requires opengl
# TODO use proper version of windows SDK (32 vs 64)
# TODO make it possible to use older versions of windows SDK (7.0+ should also work)
# TODO it windows SDK is NOT FOUND, throw ERROR
if (WIN32)
	set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/Program Files/Windows Kits/8.1/Lib/winv6.3/um/x86")
	message(" - Found windows 8.1 SDK")
	#set (CMAKE_PREFIX_PATH "C:/Program Files/Windows Kits/8.1/Lib/winv6.3/um/x64")
endif()

# Dependencies must have a version number, to ensure reproducible build. The version provided here is the one that is in the extdep repository. If you use system libraries, version numbers may be different.

find_package (CryptoPP 5.6.2 EXACT REQUIRED)
message(" - CryptoPP header: ${CRYPTOPP_INCLUDE_DIRS}")
message(" - CryptoPP lib   : ${CRYPTOPP_LIBRARIES}")

find_package (LevelDB REQUIRED)
message(" - LevelDB header: ${LEVELDB_INCLUDE_DIR}")
message(" - LevelDB lib: ${LEVELDB_LIBRARY}")

# TODO the Jsoncpp package does not yet check for correct version number
find_package (Jsoncpp 0.60 REQUIRED)
message(" - Jsoncpp header: ${JSONCPP_INCLUDE_DIRS}")
message(" - Jsoncpp lib   : ${JSONCPP_LIBRARIES}")

# TODO the JsonRpcCpp package does not yet check for correct version number
# json-rpc-cpp support is currently not mandatory
# TODO make headless client optional
# TODO get rid of -DETH_JSONRPC
find_package (JsonRpcCpp 0.3.2)
if (JSON_RPC_CPP_FOUND)
	message (" - json-rpc-cpp header: ${JSON_RPC_CPP_INCLUDE_DIRS}")
	message (" - json-rpc-cpp lib   : ${JSON_RPC_CPP_LIBRARIES}")
	add_definitions(-DETH_JSONRPC)
endif()

# TODO readline package does not yet check for correct version number
# TODO make readline package dependent on cmake options
# TODO get rid of -DETH_READLINE
find_package (Readline 6.3.8)
if (READLINE_FOUND)
	message (" - readline header: ${READLINE_INCLUDE_DIR}")
	message (" - readline lib   : ${READLINE_LIBRARY}")
	add_definitions(-DETH_READLINE)
endif ()

# TODO miniupnpc package does not yet check for correct version number
# TODO make miniupnpc package dependent on cmake options
# TODO get rid of -DMINIUPNPC
find_package (Miniupnpc 1.8.2013)
if (MINIUPNPC_FOUND)
	message (" - miniupnpc header: ${MINIUPNPC_INCLUDE_DIR}")
	message (" - miniupnpc lib   : ${MINIUPNPC_LIBRARY}")
	add_definitions(-DETH_MINIUPNPC)
endif()

# TODO gmp package does not yet check for correct version number
# TODO it is also not required in msvc build
find_package (Gmp 6.0.0)
if (GMP_FOUND)
	message(" - gmp Header: ${GMP_INCLUDE_DIR}")
	message(" - gmp lib   : ${GMP_LIBRARY}")
endif()

# curl is only requried for tests
# TODO specify min curl version, on windows we are currenly using 7.29
find_package (curl)
message(" - curl header: ${CURL_INCLUDE_DIR}")
message(" - curl lib   : ${CURL_LIBRARY}")

# TODO make headless client optional
find_package (QT5Core REQUIRED)
find_package (QT5Gui REQUIRED)
find_package (Qt5Quick REQUIRED)
find_package (Qt5Qml REQUIRED)
find_package (Qt5Network REQUIRED)
find_package (Qt5Widgets REQUIRED)
find_package (Qt5WebKit REQUIRED)
find_package (Qt5WebKitWidgets REQUIRED)

# we have to specify here if we want static and boost version, that is really important
set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)

# TODO hanlde other msvc versions or it will fail find them
if (${CMAKE_CXX_COMPILER_ID} MATCHES "MSVC")
	set(Boost_COMPILER -vc120)
endif()

find_package(Boost 1.55.0 REQUIRED COMPONENTS thread date_time system regex chrono filesystem unit_test_framework)

message(" - boost header: ${Boost_INCLUDE_DIRS}")
message(" - boost lib   : ${Boost_LIBRARIES}")

if (APPLE)
	link_directories(/usr/local/lib)
	include_directories(/usr/local/include)
endif()

