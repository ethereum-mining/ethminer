cmake_minimum_required (VERSION 2.6)

project (OPENCL_ICD_LOADER)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

set (OPENCL_ICD_LOADER_SOURCES icd.c icd_dispatch.c)

if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    list (APPEND OPENCL_ICD_LOADER_SOURCES icd_linux.c icd_exports.map)
else ()
    list (APPEND OPENCL_ICD_LOADER_SOURCES icd_windows.c OpenCL.def)
    include_directories ($ENV{DXSDK_DIR}/Include)
endif ()

# Change this to point to a directory containing OpenCL header directory "CL"
# OR copy OpenCL headers to ./inc/CL/
include_directories (./inc)

add_library (OpenCL SHARED ${OPENCL_ICD_LOADER_SOURCES})
set_target_properties (OpenCL PROPERTIES VERSION "1.2" SOVERSION "1")

if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set_target_properties (OpenCL PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,../icd_exports.map")
endif ()

target_link_libraries (OpenCL ${CMAKE_DL_LIBS})

enable_testing()
add_subdirectory (test)
