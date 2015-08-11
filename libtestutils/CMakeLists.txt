cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ${JSONCPP_INCLUDE_DIRS})
include_directories(BEFORE ..)
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE testutils)

file(GLOB HEADERS "*.h")

if (NOT JSONRPC)
	list(REMOVE_ITEM SRC_LIST "./FixedWebThreeServer.cpp")
	list(REMOVE_ITEM HEADERS "./FixedWebThreeServer.h")
endif()

add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

target_link_libraries(${EXECUTABLE} ${Boost_FILESYSTEM_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${JSONCPP_LIBRARIES})
target_link_libraries(${EXECUTABLE} ethereum)

if (JSONRPC)
	target_link_libraries(${EXECUTABLE} web3jsonrpc)
endif()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

