set(SOURCES
    wrapnvml.h wrapnvml.cpp
)

add_library(hwmon ${SOURCES})
target_link_libraries(hwmon devcore)
target_include_directories(hwmon PRIVATE ..)

if (ETHASHCUDA)
	find_package(CUDA REQUIRED)
	target_include_directories(hwmon PUBLIC ${CUDA_INCLUDE_DIRS})
endif()