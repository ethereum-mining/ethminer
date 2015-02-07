cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(..)

set(EXECUTABLE sc)

add_executable(${EXECUTABLE} ${SRC_LIST})

if (NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
target_link_libraries(${EXECUTABLE} serpent)
endif()
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} evmcore)
target_link_libraries(${EXECUTABLE} devcore)

install( TARGETS ${EXECUTABLE} DESTINATION bin )

