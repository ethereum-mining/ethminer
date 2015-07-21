cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ${JSONCPP_INCLUDE_DIRS})
include_directories(BEFORE ..)
include_directories(${DB_INCLUDE_DIRS})

set(EXECUTABLE exp)

add_executable(${EXECUTABLE} ${SRC_LIST})

target_link_libraries(${EXECUTABLE} ${Boost_REGEX_LIBRARIES})

if (READLINE_FOUND)
	target_link_libraries(${EXECUTABLE} ${READLINE_LIBRARIES})
endif()

if (JSONRPC)
	target_link_libraries(${EXECUTABLE} web3jsonrpc)
endif()

target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} p2p)
if (ETHASHCL)
#	target_link_libraries(${EXECUTABLE} ethash-cl)
#	target_link_libraries(${EXECUTABLE} ethash)
#	target_link_libraries(${EXECUTABLE} OpenCL)
endif()
target_link_libraries(${EXECUTABLE} ethcore)
install( TARGETS ${EXECUTABLE} DESTINATION bin)





