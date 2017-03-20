cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${Boost_INCLUDE_DIRS})

if (ETHASHCUDA)
	include_directories(${CUDA_INCLUDE_DIRS})
endif ()

file(GLOB HEADERS "*.h")

add_library(ethcore ${SRC_LIST} ${HEADERS})
target_link_libraries(ethcore ethash pthread)

if (ETHASHCL)
	target_link_libraries(ethcore ethash-cl)
endif ()
if (ETHASHCUDA)
	target_link_libraries(ethcore ethash-cuda)
endif ()
