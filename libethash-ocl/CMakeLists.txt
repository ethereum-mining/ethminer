# A custom command and target to turn the OpenCL kernel into a byte array header
# The normal build depends on it properly and if the kernel file is changed, then
# a rebuild of libethash-cl should be triggered

set(SOURCES
  CLMiner.h CLMiner.cpp
  AOCLUtils/opencl.cpp
)

# if(APPLE)
#     # On macOS use system OpenCL library.
#     find_package(OpenCL REQUIRED)
# else()
#     hunter_add_package(OpenCL)
#     find_package(OpenCL CONFIG REQUIRED)
# endif()

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(..)

add_library(ethash-ocl ${SOURCES})

set(CMAKE_CXX_FLAGS "-Wno-ignored-qualifiers -Wno-deprecated-declarations -Wno-unused-parameter -Wno-ignored-attributes")
target_compile_options(ethash-ocl PRIVATE ${AOCL_COMPILER_OPTS})
target_link_libraries(ethash-ocl ${AOCL_LINK_LIBRARIES})
set_target_properties(ethash-ocl PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
target_link_libraries(ethash-ocl PUBLIC ethcore ethash::ethash)
