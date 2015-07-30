set(EXECUTABLE ethash-cl)

# A custom command and target to turn the OpenCL kernel into a byte array header
# The normal build depends on it properly and if the kernel file is changed, then
# a rebuild of libethash-cl should be triggered
add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ethash_cl_miner_kernel.h
	COMMAND ${CMAKE_COMMAND} ARGS
	-DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/ethash_cl_miner_kernel.cl"
	-DBIN2H_VARIABLE_NAME=ethash_cl_miner_kernel
	-DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/ethash_cl_miner_kernel.h"
	-P "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
	COMMENT "Generating OpenCL Kernel Byte Array"
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ethash_cl_miner_kernel.cl
)
add_custom_target(clbin2h DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ethash_cl_miner_kernel.h ${CMAKE_CURRENT_SOURCE_DIR}/ethash_cl_miner_kernel.cl)

aux_source_directory(. SRC_LIST)
file(GLOB OUR_HEADERS "*.h")
set(HEADERS ${OUR_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/ethash_cl_miner_kernel.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(..)
add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
target_include_directories(${EXECUTABLE} PUBLIC ${OpenCL_INCLUDE_DIR})
target_link_libraries(${EXECUTABLE} ${OpenCL_LIBRARIES} ethash)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
