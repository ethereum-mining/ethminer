cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

set(EXECUTABLE pyserpent)

file(GLOB HEADERS "*.h") 

# set(CMAKE_INSTALL_PREFIX ../lib)
add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})

include_directories(..)

target_link_libraries(${EXECUTABLE} serpent)
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} evmcore)
target_link_libraries(${EXECUTABLE} devcore)
target_link_libraries(${EXECUTABLE} ${PYTHON_LS})

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

