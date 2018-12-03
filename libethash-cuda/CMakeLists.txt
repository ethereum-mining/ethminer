if (PROGPOW)
	add_custom_command(
		OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/progpow/CUDAMiner_kernel.h
		COMMAND ${CMAKE_COMMAND} ARGS
		-DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/progpow/CUDAMiner_kernel.cu"
		-DBIN2H_VARIABLE_NAME=CUDAMiner_kernel
		-DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/progpow/CUDAMiner_kernel.h"
		-P "libethash-cl/bin2h.cmake"
		COMMENT "Generating CUDA Kernel Byte Array"
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/progpow/CUDAMiner_kernel.cu
	)
	add_custom_target(cuda_kernel DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/progpow/CUDAMiner_kernel.h ${CMAKE_CURRENT_SOURCE_DIR}/progpow/CUDAMiner_kernel.cu)
endif()

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v;-use_fast_math)

if (NOT MSVC)
	list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
endif()

list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)
list(APPEND CUDA_NVCC_FLAGS_DEBUG -G)

if(COMPUTE AND (COMPUTE GREATER 0))
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${COMPUTE},code=sm_${COMPUTE}")
else()
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_53,code=sm_53")
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60")
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_62,code=sm_62")
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=sm_70")
	if(NOT CUDA_VERSION VERSION_LESS 10.0)
		list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75")
	endif()
endif()
if(PROGPOW)
	file(GLOB sources "progpow/*.cpp" "progpow/*.cu")
	file(GLOB headers "progpow/*.h" "progpow/*.cuh" ${CMAKE_CURRENT_BINARY_DIR}/progpow/CUDAMiner_kernel.h)
	#file(GLOB sources CUDAMiner.cpp CUDAMiner_cuda.cu)
	#file(GLOB headers CUDAMiner.h progpow/CUDAMiner_cuda.h ${CMAKE_CURRENT_BINARY_DIR}/progpow/CUDAMiner_kernel.h)	
else()
	file(GLOB sources "ethash/*.cpp" "ethash/*.cu")
	file(GLOB headers "ethash/*.h" "ethash/*.cuh")
endif()

cuda_add_library(ethash-cuda STATIC ${sources} ${headers})

if(PROGPOW)
	add_dependencies(ethash-cuda cuda_kernel)
	# Cmake doesn't handle nvrtc automatically
	find_library(CUDA_nvrtc_LIBRARY NAMES nvrtc PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib/x64 lib64/stubs lib/x64/stubs NO_DEFAULT_PATH)
	find_library(CUDA_cuda_LIBRARY NAMES cuda PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib/x64 lib64/stubs lib/x64/stubs NO_DEFAULT_PATH)
	target_link_libraries(ethash-cuda ethcore ethash::ethash progpow Boost::filesystem)
	target_link_libraries(ethash-cuda ${CUDA_nvrtc_LIBRARY} ${CUDA_cuda_LIBRARY})
else()
	target_link_libraries(ethash-cuda ethcore ethash::ethash  Boost::filesystem)
endif()
target_include_directories(ethash-cuda PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(ethash-cuda PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
