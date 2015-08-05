cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)

# we may not use it in libevm, but one of our dependecies is including boost in header file
# and windows is failing to build without that
include_directories(BEFORE ..)
include_directories(${Boost_INCLUDE_DIRS})
if (EVMJIT)
	include_directories(../evmjit/include)
endif()

set(EXECUTABLE evm)

file(GLOB HEADERS "*.h")

add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} evmcore)
target_link_libraries(${EXECUTABLE} devcore)
if (EVMJIT)
	target_link_libraries(${EXECUTABLE} evmjit-cpp)
endif()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

