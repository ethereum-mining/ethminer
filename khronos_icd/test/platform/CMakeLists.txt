add_library (IcdLog SHARED icd_test_log.c)
target_include_directories(IcdLog PUBLIC ${OpenCL_INCLUDE_DIR})