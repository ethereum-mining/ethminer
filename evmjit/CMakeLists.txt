cmake_minimum_required(VERSION 2.8.12)

project(evmjit)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_AUTOMOC OFF)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
else()
	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Wconversion -Wno-sign-conversion -Wno-unknown-pragmas ${CMAKE_CXX_FLAGS}")
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	# Do not allow unresovled symbols in shared library (default on linux)
	set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif()

# LLVM
find_package(LLVM 3.7 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(LLVM_LIBS core support mcjit x86asmparser x86codegen ipo)

get_filename_component(EVMJIT_INCLUDE_DIR include ABSOLUTE)

add_subdirectory(libevmjit)

if(EVMJIT_CPP)
	add_subdirectory(libevmjit-cpp)
endif()

if(EVMJIT_TOOLS)
	add_subdirectory(evmcc)
endif()
