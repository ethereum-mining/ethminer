project(ethereum)
cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_AUTOMOC ON)

cmake_policy(SET CMP0015 NEW)

set(HEADLESS CACHE BOOL 0)
set(LANGUAGES CACHE BOOL 0)

if ("x${PARANOIA}" STREQUAL "x")
    if ("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
		set(PARANOIA CACHE BOOL 1)
    else ()
		set(PARANOIA CACHE BOOL 0)
    endif ()
endif ()

if ("x${VMTRACE}" STREQUAL "x")
    if ("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
		set(VMTRACE CACHE BOOL 1)
    else ()
		set(VMTRACE CACHE BOOL 0)
    endif ()
endif ()

if (PARANOIA)
	add_definitions(-DETH_PARANOIA)
endif ()

if (VMTRACE)
	add_definitions(-DETH_VMTRACE)
endif ()

if (LANGUAGES)
	add_definitions(-DETH_LANGUAGES)
endif ()

# Default TARGET_PLATFORM to "linux".
set(TARGET_PLATFORM CACHE STRING "linux")
if ("x${TARGET_PLATFORM}" STREQUAL "x")
	set(TARGET_PLATFORM "linux")
endif ()

if ("${TARGET_PLATFORM}" STREQUAL "linux")
	set(CMAKE_THREAD_LIBS_INIT pthread)
endif ()

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS                "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -DSHAREDLIB")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DETH_DEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG -DETH_RELEASE")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG -DETH_RELEASE")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DETH_DEBUG")

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

if (CMAKE_COMPILER_IS_MINGW)
	set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/mingw")
elseif (CMAKE_COMPILER_IS_MSYS)
	set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/msys")
elseif (CMAKE_COMPILER_IS_GNUCXX)
	set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/g++")
elseif (CMAKE_COMPILER_IS_MSVC)
	set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/msvc")
else ()
	set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/unknown")
endif ()


message("CXXFLAGS: ${CMAKE_CXX_FLAGS}")
#add_definitions("-DETH_BUILD_TYPE=${ETH_BUILD_TYPE}")
#add_definitions("-DETH_BUILD_PLATFORM=${ETH_BUILD_PLATFORM}")

# Compiler-specific C++11 activation.
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

if(${TARGET_PLATFORM} STREQUAL "w64")
#	set(MINIUPNPC_LS /usr/x86_64-w64-mingw32/lib/libminiupnpc.a)
	set(LEVELDB_LS leveldb)
	set(CRYPTOPP_LS cryptopp)
	set(CRYPTOPP_ID /usr/x86_64-w64-mingw32/include/cryptopp)
else()
	# Look for available Crypto++ version and if it is >= 5.6.2
	find_path(CRYPTOPP_ID cryptlib.h
		../cryptopp/src
		../../cryptopp/src)
	if(NOT ${CRYPTOPP_ID} STREQUAL "CRYPTOPP_ID-NOTFOUND")
		message("Local cryptopp OK.")
		set(CRYPTOPP_LS ${CRYPTOPP_ID}/../target/build/release/libcryptopp.so)
	else()
		message("Trying system cryptopp...")
		find_path(CRYPTOPP_ID cryptlib.h
			/usr/include/cryptopp
			/usr/include/crypto++
			/usr/local/include/cryptopp
			/usr/local/include/crypto++
			/opt/local/include/cryptopp
			/opt/local/include/crypto++
			)
		find_library(CRYPTOPP_LS NAMES cryptoppeth cryptopp
			PATHS
			/usr/lib
			/usr/local/lib
			/opt/local/lib
			)
	endif()

	if (CRYPTOPP_ID AND CRYPTOPP_LS)
		message(STATUS "Found Crypto++: ${CRYPTOPP_ID}, ${CRYPTOPP_LS}")
		set(_CRYPTOPP_VERSION_HEADER ${CRYPTOPP_ID}/config.h)
		if(EXISTS ${_CRYPTOPP_VERSION_HEADER})
			file(STRINGS ${_CRYPTOPP_VERSION_HEADER} _CRYPTOPP_VERSION REGEX "^#define CRYPTOPP_VERSION[ \t]+[0-9]+$")
			string(REGEX REPLACE "^#define CRYPTOPP_VERSION[ \t]+([0-9]+)" "\\1" _CRYPTOPP_VERSION ${_CRYPTOPP_VERSION})
			if(${_CRYPTOPP_VERSION} LESS 562)
				set(CRYPTOPP_ID CACHE FILEPATH "" FORCE)
				set(CRYPTOPP_LS CACHE FILEPATH "" FORCE)
				message(STATUS "Crypto++ version found is smaller than 5.6.2.")
			else()
				message(STATUS "Crypto++ found and version greater or equal to 5.6.2")
			endif()
		endif()
	endif()

	# Not really worth caching. We want to reevaluate anyway.
	mark_as_advanced(CRYPTOPP_ID CRYPTOPP_LS)

	find_path( LEVELDB_ID leveldb/db.h
		/usr/include
		/usr/local/include
		)
	if ( LEVELDB_ID )
		message(STATUS "Found LevelDB Headers")

		# Check for accessory dev libraries leveldb and miniupnpc
		find_library( LEVELDB_LS NAMES leveldb
			PATHS
			/usr/lib
			/usr/local/lib
			/opt/local/lib
			/usr/lib/*/
			)
		if ( LEVELDB_LS )
			message(STATUS "Found LevelDB Library: ${LEVELDB_LS}")
			add_definitions(-DETH_LEVELDB)
		else ()
			message(FATAL_ERROR "Failed to find the LevelDB Library!")
		endif ()
	else ()
		message(FATAL_ERROR "Failed to find the LevelDB headers")
	endif ()

	find_path( PYTHON_ID pyconfig.h
		/usr/include/python2.7
		/usr/local/include/python2.7
		)
	if ( PYTHON_ID )
		message(STATUS "Found Python-2.7 Headers: ${PYTHON_ID}")

		# Check for accessory dev libraries leveldb and miniupnpc
		find_library( PYTHON_LS NAMES python2.7
			PATHS
			/usr/lib
			/usr/local/lib
			/opt/local/lib
			/usr/lib/*/
			)
		if ( PYTHON_LS )
			message(STATUS "Found Python-2.7 Library: ${PYTHON_LS}")
			add_definitions(-DETH_PYTHON)
		else ()
			message(FATAL_ERROR "Failed to find the Python-2.7 Library!")
		endif ()
	else ()
		message(FATAL_ERROR "Failed to find the Python-2.7 headers")
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
		find_package(Boost 1.53 REQUIRED COMPONENTS thread date_time system)
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

#	if (NOT APPLE)
#		link_directories(${CRYPTOPP_LS})
#		link_directories(${LEVELDB_LS})
#		link_directories(${MINIUPNPC_LS})
#		link_directories(${PYTHON_LS})
#		link_directories(${READLINE_LS})
#		link_directories(${JSONRPC_LS})
#	endif()

# Generate header file containing useful build information
add_custom_target(BuildInfo.h ALL COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/BuildInfo.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BUILD_TYPE} ${ETH_BUILD_PLATFORM})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	link_directories(/usr/local/lib)
	include_directories(/usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

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
	add_subdirectory(libqethereum)
	add_subdirectory(alethzero)
if(QTQML)
	add_subdirectory(walleth)
endif()
endif()
endif()

enable_testing()
add_test(NAME alltests WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test COMMAND testeth)

#unset(HEADLESS CACHE)
#unset(TARGET_PLATFORM CACHE)

