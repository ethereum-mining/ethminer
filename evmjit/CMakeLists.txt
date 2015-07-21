cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW) 		# allow VERSION argument in project()

project(EVMJIT VERSION 0.9.0 LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_AUTOMOC OFF)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
else()
	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Wconversion -Wno-sign-conversion -Wno-unknown-pragmas ${CMAKE_CXX_FLAGS}")
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT ${CMAKE_BUILD_TYPE} STREQUAL "DebugSan")
	# Do not allow unresovled symbols in shared library (default on linux)
	set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif()

# LLVM
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT LLVM_DIR)
	# Workaround for Ubuntu broken LLVM package
	message(STATUS "Using llvm-3.7-dev package from Ubuntu. If does not work, build LLVM and set -DLLVM_DIR=llvm-build/share/llvm/cmake")
	execute_process(COMMAND llvm-config-3.7 --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIRS)
	message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
	set(LLVM_LIBS "-lLLVMipo -lLLVMVectorize -lLLVMX86AsmParser -lLLVMX86CodeGen -lLLVMX86Desc -lLLVMX86Info -lLLVMMCDisassembler -lLLVMX86AsmPrinter -lLLVMX86Utils -lLLVMSelectionDAG -lLLVMAsmPrinter -lLLVMCodeGen -lLLVMScalarOpts -lLLVMProfileData -lLLVMInstCombine -lLLVMInstrumentation -lLLVMTransformUtils -lLLVMipa -lLLVMMCJIT -lLLVMExecutionEngine -lLLVMTarget -lLLVMAnalysis -lLLVMRuntimeDyld -lLLVMObject -lLLVMMCParser -lLLVMBitReader -lLLVMMC -lLLVMCore -lLLVMSupport -lz -lpthread -lffi -ltinfo -ldl -lm")
	add_definitions(-D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS)
	link_directories(/usr/lib/llvm-3.7/lib)
else()
	find_package(LLVM 3.7 REQUIRED CONFIG)
	message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
	message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
	add_definitions(${LLVM_DEFINITIONS})
	llvm_map_components_to_libnames(LLVM_LIBS core support mcjit x86asmparser x86codegen ipo)
endif()

get_filename_component(EVMJIT_INCLUDE_DIR include ABSOLUTE)

add_subdirectory(libevmjit)

if(EVMJIT_CPP)
	add_subdirectory(libevmjit-cpp)
endif()
