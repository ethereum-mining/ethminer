set(EXECUTABLE ethash-cl)

include(bin2h.cmake)
bin2h(SOURCE_FILE ethash_cl_miner_kernel.cl
	VARIABLE_NAME ethash_cl_miner_kernel
	HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/ethash_cl_miner_kernel.h)

aux_source_directory(. SRC_LIST)
file(GLOB HEADERS "*.h")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(..)
add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${OpenCL_LIBRARIES} ethash)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

