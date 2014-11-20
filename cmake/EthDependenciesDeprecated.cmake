# search for and configure dependencies

# deprecated. DO NOT ADD any new stuff here. Proper dependency fetching is done in EthDependencies.cmake 


if("${TARGET_PLATFORM}" STREQUAL "w64")
#	set(MINIUPNPC_LS /usr/x86_64-w64-mingw32/lib/libminiupnpc.a)
	set(LEVELDB_LS leveldb)
else()

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




if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	link_directories(/usr/local/lib)
	include_directories(/usr/local/include)
endif()
