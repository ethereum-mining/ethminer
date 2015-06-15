cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)

set(EXECUTABLE ethereumx)

file(GLOB HEADERS "*.h") 

add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

target_link_libraries(${EXECUTABLE} ${LEVELDB_LS})
target_link_libraries(${EXECUTABLE} ${CRYPTOPP_LS})

if (MINIUPNPC_LS)
	target_link_libraries(${EXECUTABLE} ${MINIUPNPC_LS})
endif()

target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} evm)
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} secp256k1)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

