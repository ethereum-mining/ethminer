cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${DB_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE whisper)

file(GLOB HEADERS "*.h")

add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

#target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} p2p)
target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} devcore)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
