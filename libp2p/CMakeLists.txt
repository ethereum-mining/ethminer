cmake_policy(SET CMP0015 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	# old policy do not use MACOSX_RPATH
	cmake_policy(SET CMP0042 OLD)
endif()
set(CMAKE_AUTOMOC OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
# we may not use it in libp2p, but one of our dependecies is including leveldb in header file
# and windows is failing to build without that
include_directories(${DB_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

if (MINIUPNPC_FOUND)
	include_directories(${MINIUPNPC_INCLUDE_DIRS})
endif()

set(EXECUTABLE p2p)

file(GLOB HEADERS "*.h")

add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

if (MINIUPNPC_FOUND)
	target_link_libraries(${EXECUTABLE} ${MINIUPNPC_LIBRARIES})
endif()

target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} devcore)

if (CMAKE_COMPILER_IS_MINGW)
	target_link_libraries(${EXECUTABLE} ws2_32 mswsock)
endif()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

