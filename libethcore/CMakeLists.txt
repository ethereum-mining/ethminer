cmake_policy(SET CMP0015 NEW)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

include_directories(BEFORE ..)
include_directories(${Boost_INCLUDE_DIRS})

if (ETHASHCUDA)
	include_directories(${CUDA_INCLUDE_DIRS})
endif ()

add_library(ethcore ${SOURCES} ${HEADERS})
target_link_libraries(ethcore ethash pthread)

if (ETHASHCL)
	target_link_libraries(ethcore ethash-cl)
endif ()
if (ETHASHCUDA)
	target_link_libraries(ethcore ethash-cuda)
endif ()
