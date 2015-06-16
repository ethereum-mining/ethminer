set(EXECUTABLE ethash-cl)

include(bin2h.cmake)
bin2h(SOURCE_FILE ethash_cl_miner_kernel.cl
	VARIABLE_NAME ethash_cl_miner_kernel
	HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/ethash_cl_miner_kernel.h)

#   Add a custom command so that the user can regenerate the header file containing the kernel
#   code's bytearray by running "make clbin2h"
	add_custom_target(clbin2h)
	add_custom_command(TARGET clbin2h
	PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/ethash_cl_miner_kernel.cl"
	-DBIN2H_VARIABLE_NAME=ethash_cl_miner_kernel
	-DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/ethash_cl_miner_kernel.h"
	-P "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake")

aux_source_directory(. SRC_LIST)
file(GLOB HEADERS "*.h")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(..)
add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${OpenCL_LIBRARIES} ethash)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

