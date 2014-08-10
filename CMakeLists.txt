# cmake global
project(ethereum)
cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_AUTOMOC ON)
cmake_policy(SET CMP0015 NEW)

# user defined, defaults
# Normally, set(...CACHE...) creates cache variables, but does not modify them.

set(HEADLESS OFF CACHE BOOL "Do not compile GUI (AlethZero)")
set(LANGUAGES OFF CACHE BOOL "Limit build to Serpent/LLL tools")
set(VMTRACE OFF CACHE BOOL "VM tracing and run-time checks (useful for cross-implementation VM debugging)")
set(PARANOIA OFF CACHE BOOL "Additional run-time checks")

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

message("LANGUAGES: ${LANGUAGES}; VMTRACE: ${VMTRACE}; PARANOIA: ${PARANOIA}; HEADLESS: ${HEADLESS}")

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

# Initialize CXXFLAGS
set(CMAKE_CXX_FLAGS                "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -DSHAREDLIB")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DETH_DEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG -DETH_RELEASE")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG -DETH_RELEASE")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DETH_DEBUG")

# Windows 
if ("${TARGET_PLATFORM}" STREQUAL "w64")
	set(CMAKE_SYSTEM_NAME Windows)

	set(CMAKE_CXX_LIBRARY_ARCHITECTURE	x86_64-w64-mingw32)
	set(CMAKE_C_COMPILER			x86_64-w64-mingw32-gcc)
	set(CMAKE_CXX_COMPILER			x86_64-w64-mingw32-g++)
	set(CMAKE_RC_COMPILER			x86_64-w64-mingw32-windres)
	set(CMAKE_AR				x86_64-w64-mingw32-ar)
	set(CMAKE_RANLIB			x86_64-w64-mingw32-ranlib)

	set(CMAKE_EXECUTABLE_SUFFIX		.exe)

	set(CMAKE_FIND_ROOT_PATH
		/usr/x86_64-w64-mingw32
	)

	include_directories(/usr/x86_64-w64-mingw32/include/cryptopp)

	set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
	set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
	set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

	set(CMAKE_INSTALL_PREFIX	/usr/x86_64-w64-mingw32)
	set(ETH_BUILD_PLATFORM "windows")
	set(ETH_STATIC 1)
else ()
	set(ETH_BUILD_PLATFORM ${CMAKE_SYSTEM_NAME})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
	set(ETH_SHARED 1)
endif()

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

message("CXXFLAGS: ${CMAKE_CXX_FLAGS}")
#add_definitions("-DETH_BUILD_TYPE=${ETH_BUILD_TYPE}")
#add_definitions("-DETH_BUILD_PLATFORM=${ETH_BUILD_PLATFORM}")

# C++11 check and activation
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

if("${TARGET_PLATFORM}" STREQUAL "w64")
#	set(MINIUPNPC_LS /usr/x86_64-w64-mingw32/lib/libminiupnpc.a)
	set(LEVELDB_LS leveldb)
	set(CRYPTOPP_LS cryptopp)
	set(CRYPTOPP_ID /usr/x86_64-w64-mingw32/include/cryptopp)
else()
	# Look for available Crypto++ version and if it is >= 5.6.2
	find_path(ID cryptlib.h
		../cryptopp/src
		../../cryptopp/src
		/usr/include/cryptopp
		/usr/include/crypto++
		/usr/local/include/cryptopp
		/usr/local/include/crypto++
		/opt/local/include/cryptopp
		/opt/local/include/crypto++
		)
	find_library(LS NAMES cryptoppeth cryptopp
		../cryptopp/src/../target/build/release
		../../cryptopp/src/../target/build/release
		PATHS
		/usr/lib
		/usr/local/lib
		/opt/local/lib
		)

	if (ID AND LS)
		message(STATUS "Found Crypto++: ${ID}, ${LS}")
		set(_CRYPTOPP_VERSION_HEADER ${ID}/config.h)
		if(EXISTS ${_CRYPTOPP_VERSION_HEADER})
			file(STRINGS ${_CRYPTOPP_VERSION_HEADER} _CRYPTOPP_VERSION REGEX "^#define CRYPTOPP_VERSION[ \t]+[0-9]+$")
			string(REGEX REPLACE "^#define CRYPTOPP_VERSION[ \t]+([0-9]+)" "\\1" _CRYPTOPP_VERSION ${_CRYPTOPP_VERSION})
			if(${_CRYPTOPP_VERSION} LESS 562)
				message(FATAL_ERROR "Crypto++ version found is smaller than 5.6.2.")
			else()
				set(CRYPTOPP_ID ${ID} CACHE FILEPATH "")
				set(CRYPTOPP_LS ${LS} CACHE FILEPATH "")
				message(STATUS "Crypto++ found and version greater or equal to 5.6.2")
			endif()
		endif()
  else()
    message(STATUS "Crypto++ Not Found: ${CRYPTOPP_ID}, ${CRYPTOPP_LS}")
	endif()

	find_path( LEVELDB_ID leveldb/db.h
		/usr/include
		/usr/local/include
		)
	if ( LEVELDB_ID STREQUAL "LEVELDB_ID-NOTFOUND" )
		message(FATAL_ERROR "Failed to find the LevelDB headers")
	else ()
		message(STATUS "Found LevelDB Headers")

		# Check for accessory dev libraries leveldb and miniupnpc
		find_library( LEVELDB_LS NAMES leveldb
			PATHS
			/usr/lib
			/usr/local/lib
			/opt/local/lib
			/usr/lib/*/
			)
		if ( LEVELDB_LS STREQUAL "LEVELDB_LS-NOTFOUND" )
			message(FATAL_ERROR "Failed to find the LevelDB Library!")
		else ()
			message(STATUS "Found LevelDB Library: ${LEVELDB_LS}")
			add_definitions(-DETH_LEVELDB)
		endif ()
	endif ()

	find_path( PYTHON_ID pyconfig.h
		${PYTHON_INCLUDE_DIR}
		/usr/include/python2.7
		/usr/local/include/python2.7
		)
	if ( PYTHON_ID STREQUAL "PYTHON_ID-NOTFOUND" )
		message(STATUS "Failed to find the Python-2.7 headers")
	else ()
		message(STATUS "Found Python-2.7 Headers: ${PYTHON_ID}")

		# Check for accessory dev libraries leveldb and miniupnpc
		find_library( PYTHON_LS NAMES python2.7
			PATHS
			/usr/lib
			/usr/local/lib
			/opt/local/lib
			/usr/lib/*/
			)
		if ( PYTHON_LS STREQUAL "PYTHON_LS-NOTFOUND" )
			message(STATUS "Failed to find the Python-2.7 Library!")
			set(PYTHON_ID)
			set(PYTHON_LS)
		else ()
			message(STATUS "Found Python-2.7 Library: ${PYTHON_LS}")
			add_definitions(-DETH_PYTHON)
		endif ()
	endif ()

	find_path( MINIUPNPC_ID miniupnpc/miniwget.h
		/usr/include
		/usr/local/include
		)
	if ( MINIUPNPC_ID ) 
		message(STATUS "Found miniupnpc headers")

		find_library( MINIUPNPC_LS NAMES miniupnpc
			PATHS
			/usr/lib
			/usr/local/lib
			/opt/local/lib
			/usr/lib/*/
			)
		if ( MINIUPNPC_LS )
			message(STATUS "Found miniupnpc library: ${MINIUPNPC_LS}")
			add_definitions(-DETH_MINIUPNPC)
		else ()
			message(STATUS "Failed to find the miniupnpc library!")
		endif ()
	else ()
		message(STATUS "Failed to find the miniupnpc headers!")
	endif ()

	find_path( JSONRPC_ID jsonrpc/rpc.h
		/usr/include
		/usr/local/include
		)
	if ( JSONRPC_ID )
		message(STATUS "Found jsonrpc headers")
		find_library( JSONRPC_LS NAMES jsonrpc
			PATHS
			/usr/lib
			/usr/local/lib
			/opt/local/lib
			/usr/lib/*/
			)
		if ( JSONRPC_LS )
			message(STATUS "Found jsonrpc library: ${JSONRPC_LS}")
		add_definitions(-DETH_JSONRPC)
		else ()
			message(STATUS "Failed to find the jsonrpc library!")
		endif ()
	else ()
		message(STATUS "Failed to find the jsonrpc headers!")
	endif ()

	find_path( READLINE_ID readline/readline.h
		/usr/include
		/usr/local/include
		)
	if ( READLINE_ID )
		message(STATUS "Found readline headers")
		find_library( READLINE_LS NAMES readline
			PATHS
			/usr/lib
			/usr/local/lib
			/opt/local/lib
			/usr/lib/*/
			)
		if ( READLINE_LS )
			message(STATUS "Found readline library: ${READLINE_LS}")
			add_definitions(-DETH_READLINE)
		else ()
			message(STATUS "Failed to find the readline library!")
		endif ()
	else ()
		message(STATUS "Failed to find the readline headers!")
	endif ()

	if (LANGUAGES)
		find_package(Boost 1.53 REQUIRED COMPONENTS thread date_time)
	else()
		find_package(Boost 1.53 REQUIRED COMPONENTS thread date_time system regex)
	endif()

	set(QTQML 1)
endif()

if(CRYPTOPP_ID)
	include_directories(${CRYPTOPP_ID})
endif()
if(PYTHON_ID)
	include_directories(${PYTHON_ID})
endif()
if(MINIUPNPC_ID)
	include_directories(${MINIUPNPC_ID})
endif()
if(LEVELDB_ID)
	include_directories(${LEVELDB_ID})
endif()
if(READLINE_ID)
	include_directories(${READLINE_ID})
endif()
if(JSONRPC_ID)
	include_directories(${JSONRPC_ID})
endif()

# Generate header file containing useful build information
add_custom_target(BuildInfo.h ALL COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/BuildInfo.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BUILD_TYPE} ${ETH_BUILD_PLATFORM})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	link_directories(/usr/local/lib)
	include_directories(/usr/local/include)
endif()

add_subdirectory(libethential)
add_subdirectory(libevmface)
add_subdirectory(liblll)
add_subdirectory(libserpent)
if(NOT APPLE)
if(PYTHON_LS)
add_subdirectory(libpyserpent)
endif()
endif()
add_subdirectory(lllc)
add_subdirectory(sc)
if (NOT LANGUAGES)
	add_subdirectory(secp256k1)
	add_subdirectory(libethcore)
	add_subdirectory(libevm)
	add_subdirectory(libethereum)
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
			add_subdirectory(walleth)
		endif()
	endif()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(SRC_LIST BuildInfo.h)

enable_testing()
add_test(NAME alltests WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test COMMAND testeth)

#unset(TARGET_PLATFORM CACHE)

