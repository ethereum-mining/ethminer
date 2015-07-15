cmake_minimum_required (VERSION 2.6)
cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)

project (OPENCL_ICD_LOADER)

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

add_library (OpenCL_ICD SHARED ${OPENCL_ICD_LOADER_SOURCES})
set_target_properties (OpenCL_ICD PROPERTIES VERSION "1.2" SOVERSION "1")

if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set_target_properties (OpenCL_ICD PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,../../khronos_icd/icd_exports.map")
endif ()

target_link_libraries (OpenCL_ICD ${CMAKE_DL_LIBS})
target_include_directories(OpenCL_ICD PUBLIC ${OpenCL_INCLUDE_DIR})

install(TARGETS OpenCL_ICD RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
