project(ethereum)
cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_AUTOMOC ON)

cmake_policy(SET CMP0015 NEW)

# Default HEADLESS to 0.
set(HEADLESS CACHE BOOL 0)
#if ("x${HEADLESS}" STREQUAL "x")
#	set(HEADLESS 0)
#endif ()

if ("x${PARANOIA}" STREQUAL "x")
    if ("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
        set(PARANOIA 1)
    else ()
        set(PARANOIA 0)
    endif ()
endif ()

if (PARANOIA)
    add_definitions(-DETH_PARANOIA)
endif ()

if ("x${VMTRACE}" STREQUAL "x")
    if ("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
        set(VMTRACE 1)
    else ()
        set(VMTRACE 0)
    endif ()
endif ()

if (VMTRACE)
    add_definitions(-DETH_VMTRACE)
endif ()

# Default TARGET_PLATFORM to "linux".
set(TARGET_PLATFORM CACHE STRING "linux")
if ("x${TARGET_PLATFORM}" STREQUAL "x")
	set(TARGET_PLATFORM "linux")
endif ()

if ("${TARGET_PLATFORM}" STREQUAL "linux")
	set(CMAKE_THREAD_LIBS_INIT pthread)
endif ()

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
else ()
	set(ETH_BUILD_PLATFORM ${CMAKE_SYSTEM_NAME})
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

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS                "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

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

if (${TARGET_PLATFORM} STREQUAL "w64")
else ()
	# Look for available Crypto++ version and if it is >= 5.6.2
	find_path(ID cryptlib.h
		/usr/include/cryptopp
		/usr/include/crypto++
		/usr/local/include/cryptopp
		/usr/local/include/crypto++
		/opt/local/include/cryptopp
		/opt/local/include/crypto++
		)
	find_library(LS NAMES cryptoppeth cryptopp
		PATHS
		/usr/lib
		/usr/local/lib
		/opt/local/lib
		)

	if(ID AND LS)
		message(STATUS "Found Crypto++: ${ID}, ${LS}")
		set(_CRYPTOPP_VERSION_HEADER ${ID}/config.h)
		if(EXISTS ${_CRYPTOPP_VERSION_HEADER})
			file(STRINGS ${_CRYPTOPP_VERSION_HEADER} _CRYPTOPP_VERSION REGEX "^#define CRYPTOPP_VERSION[ \t]+[0-9]+$")
			string(REGEX REPLACE "^#define CRYPTOPP_VERSION[ \t]+([0-9]+)" "\\1" _CRYPTOPP_VERSION ${_CRYPTOPP_VERSION})
			if(${_CRYPTOPP_VERSION} LESS 562)
				message(STATUS "System Crypto++ version found is smaller than 5.6.2.")
			else()
				set(CRYPTOPP_INCLUDE_DIR ${ID} CACHE FILEPATH "" FORCE)
				set(CRYPTOPP_LIBRARIES ${LS} CACHE FILEPATH "" FORCE)
				set(CRYPTOPP_FOUND TRUE)
				message(STATUS "System Crypto++ found and version greater or equal to 5.6.2")
			endif()
		endif()
	endif()

	if(NOT CRYPTOPP_FOUND)
		set(CRYPTOPP_INCLUDE_DIR "../cryptopp562" CACHE FILEPATH "" FORCE)
		find_library(LSLOC NAMES cryptoppeth cryptopp
			PATHS ../cryptopp562
			NO_DEFAULT_PATH
			)
		set(CRYPTOPP_LIBRARIES ${LSLOC} CACHE FILEPATH "" FORCE)
		message(STATUS "System Crypto++ not found, broken or too old. We use ${LSLOC}")
	endif()

	# Not really worth caching. We want to reevaluate anyway.
	mark_as_advanced(CRYPTOPP_INCLUDE_DIR CRYPTOPP_LIBRARIES)

	# Always "found", given last block.
	include_directories(${CRYPTOPP_INCLUDE_DIR})

	if(NOT APPLE)
		link_directories(${CRYPTOPP_LIBRARIES})
	endif()

	find_path( LEVELDB_ID leveldb/db.h
		/usr/include
		/usr/local/include
		)
	if ( LEVELDB_ID )
		message(STATUS "Found LevelDB Headers")
	else ()
		message(FATAL_ERROR "Failed to find the LevelDB headers")
	endif ()

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
	else () 
		message(FATAL_ERROR "Failed to find the LevelDB Library!")
	endif ()

	find_path( MINIUPNPC_ID miniupnpc/miniwget.h
		/usr/include
		/usr/local/include
		)
	if ( MINIUPNPC_ID ) 
		message(STATUS "Found miniupnpc headers")
	else () 
		message(FATAL_ERROR "Failed to find the miniupnpc headers!")
	endif ()

	find_library( MINIUPNPC_LS NAMES miniupnpc
		PATHS
		/usr/lib
		/usr/local/lib
		/opt/local/lib
		/usr/lib/*/
		)
	if ( MINIUPNPC_LS ) 
		message(STATUS "Found miniupnpc library: ${MINIUPNPC_LS}")
	else ()
		message(FATAL_ERROR "Failed to find the miniupnpc library!")
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
		else ()
			message(STATUS "Failed to find the readline library!")
		endif ()
	else ()
		message(STATUS "Failed to find the readline headers!")
	endif ()

endif()

# Generate header file containing useful build information
add_custom_target(BuildInfo.h ALL COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/BuildInfo.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BUILD_TYPE} ${ETH_BUILD_PLATFORM})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	link_directories(/usr/local/lib)
	include_directories(/usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_subdirectory(secp256k1)
add_subdirectory(libethsupport)
add_subdirectory(libethcore)
add_subdirectory(liblll)
add_subdirectory(libevm)
add_subdirectory(libethereum)
add_subdirectory(test)
add_subdirectory(lllc)
add_subdirectory(eth)
if ("x${CMAKE_BUILD_TYPE}" STREQUAL "xDebug")
	add_subdirectory(exp)
endif ()
if (NOT ("${TARGET_PLATFORM}" STREQUAL "w64"))
	add_subdirectory(neth)
endif ()

if (NOT HEADLESS)
	add_subdirectory(libqethereum)
	add_subdirectory(alethzero)
	add_subdirectory(walleth)
endif ()

enable_testing()
add_test(NAME alltests WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test COMMAND testeth)

#unset(HEADLESS CACHE)
#unset(TARGET_PLATFORM CACHE)

