cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${LEVELDB_INCLUDE_DIRS})

set(EXECUTABLE exp)

add_executable(${EXECUTABLE} ${SRC_LIST})

target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} p2p)

install( TARGETS ${EXECUTABLE} DESTINATION bin) 

