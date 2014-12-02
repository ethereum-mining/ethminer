cmake_policy(SET CMP0015 OLD)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(${LEVELDB_INCLUDE_DIR})
include_directories(..)

set(EXECUTABLE web3jsonrpc)

file(GLOB HEADERS "*.h")

if (ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})
endif()

target_link_libraries(${EXECUTABLE} ${LEVELDB_LIBRARY})
target_link_libraries(${EXECUTABLE} ${JSONCPP_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_COMMON_LIBRARY})
target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_SERVER_LIBRARY})

if (MINIUPNPC_LS)
	target_link_libraries(${EXECUTABLE} ${MINIUPNPC_LS})
endif()

target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} secp256k1)
target_link_libraries(${EXECUTABLE} gmp)
target_link_libraries(${EXECUTABLE} solidity)
target_link_libraries(${EXECUTABLE} serpent)

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )


