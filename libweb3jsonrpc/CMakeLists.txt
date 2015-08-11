cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ${JSONCPP_INCLUDE_DIRS})
include_directories(BEFORE ..)
include_directories(${MHD_INCLUDE_DIRS})
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(${DB_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE web3jsonrpc)

file(GLOB HEADERS "*.h")

add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

target_link_libraries(${EXECUTABLE} ${JSONCPP_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_SERVER_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${MHD_LIBRARIES})

target_link_libraries(${EXECUTABLE} webthree)

if (SOLIDITY)
	target_link_libraries(${EXECUTABLE} solidity)
endif ()
if (SERPENT)
	target_link_libraries(${EXECUTABLE} serpent)
endif ()

if (ETH_JSON_RPC_STUB)
	add_custom_target(jsonrpcstub) 
	add_custom_command(TARGET jsonrpcstub
		POST_BUILD
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMAND ${CMAKE_COMMAND} -DETH_SPEC_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spec.json" -DETH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
			-DETH_SERVER_DIR="${CMAKE_CURRENT_SOURCE_DIR}" -DETH_CLIENT_DIR="${CMAKE_SOURCE_DIR}/test/libweb3jsonrpc"
			-DETH_SERVER_NAME=AbstractWebThreeStubServer -DETH_CLIENT_NAME=WebThreeStubClient
			-DETH_JSON_RPC_STUB="${ETH_JSON_RPC_STUB}"
			-P "${ETH_SCRIPTS_DIR}/jsonrpcstub.cmake"
	)
	add_dependencies(${EXECUTABLE} jsonrpcstub)
endif ()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

add_custom_target(aux_json SOURCES "spec.json")
