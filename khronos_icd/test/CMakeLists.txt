include_directories (./inc)

add_subdirectory (platform)
add_subdirectory (driver_stub)
add_subdirectory (loader_test)

add_test (OPENCL_ICD_LOADER_TEST ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/icd_loader_test)
