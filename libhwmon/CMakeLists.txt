set(SOURCES
    wraphelper.cpp wraphelper.h
    wrapnvml.h wrapnvml.cpp
    wrapadl.h wrapadl.cpp
    wrapamdsysfs.h wrapamdsysfs.cpp
)

add_library(hwmon ${SOURCES})
set(HWMON_LINK_LIBRARIES devcore)

if (UNIX)
	list(APPEND HWMON_LINK_LIBRARIES dl)
endif ()

target_link_libraries(hwmon ${HWMON_LINK_LIBRARIES})
target_include_directories(hwmon PRIVATE ..)

if (ETHASHCUDA)
	find_package(CUDA REQUIRED)
	target_include_directories(hwmon PUBLIC ${CUDA_INCLUDE_DIRS})
endif()
