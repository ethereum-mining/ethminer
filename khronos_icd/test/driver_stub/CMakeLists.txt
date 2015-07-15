
set (OPENCL_DRIVER_STUB_SOURCES cl.c cl_ext.c cl_gl.c icd.c)

if (NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    list (APPEND OPENCL_DRIVER_STUB_SOURCES driver_stub.def)
endif ()

add_library (OpenCLDriverStub SHARED ${OPENCL_DRIVER_STUB_SOURCES})

target_link_libraries (OpenCLDriverStub IcdLog)
