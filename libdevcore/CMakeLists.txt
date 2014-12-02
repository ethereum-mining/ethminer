cmake_policy(SET CMP0015 OLD)
set(CMAKE_AUTOMOC OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)

include_directories(${LEVELDB_INCLUDE_DIR})
include_directories(..)

set(EXECUTABLE devcore)

file(GLOB HEADERS "*.h")

if(ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})
endif()

if (APPLE)
	# Latest mavericks boost libraries only come with -mt
	target_link_libraries(${EXECUTABLE} boost_system-mt)
	target_link_libraries(${EXECUTABLE} boost_filesystem-mt)
	target_link_libraries(${EXECUTABLE} boost_thread-mt)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
elseif (UNIX)
	find_package(Boost 1.53 REQUIRED COMPONENTS thread system)
	target_link_libraries(${EXECUTABLE} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY})
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
endif()

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

