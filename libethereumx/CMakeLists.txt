cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)
include_directories(..)

set(EXECUTABLE ethereumx)

if(ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${SRC_LIST})
else()
	add_library(${EXECUTABLE} SHARED ${SRC_LIST})
endif()

file(GLOB HEADERS "*.h") 


target_link_libraries(${EXECUTABLE} ${LEVELDB_LS})
target_link_libraries(${EXECUTABLE} ${CRYPTOPP_LS})

if (MINIUPNPC_LS)
target_link_libraries(${EXECUTABLE} ${MINIUPNPC_LS})
endif()

target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} evm)
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} secp256k1)
target_link_libraries(${EXECUTABLE} gmp)

if (APPLE)
	# Latest mavericks boost libraries only come with -mt
	target_link_libraries(${EXECUTABLE} boost_system-mt)
	target_link_libraries(${EXECUTABLE} boost_regex-mt)
	target_link_libraries(${EXECUTABLE} boost_filesystem-mt)
	target_link_libraries(${EXECUTABLE} boost_thread-mt)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
elseif (UNIX)
	target_link_libraries(${EXECUTABLE} ${Boost_SYSTEM_LIBRARY})
	target_link_libraries(${EXECUTABLE} ${Boost_REGEX_LIBRARY})
	target_link_libraries(${EXECUTABLE} ${Boost_FILESYSTEM_LIBRARY})
	target_link_libraries(${EXECUTABLE} ${Boost_THREAD_LIBRARY})
	target_link_libraries(${EXECUTABLE} ${Boost_DATE_TIME_LIBRARY})
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
endif ()

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

