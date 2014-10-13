# search for and configure dependencies

# deprecated. TODO will rewrite to proper CMake packages



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




if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	link_directories(/usr/local/lib)
	include_directories(/usr/local/include)
endif()
