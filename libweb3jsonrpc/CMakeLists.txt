cmake_policy(SET CMP0015 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	# old policy do not use MACOSX_RPATH
	cmake_policy(SET CMP0042 OLD)
endif()
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(${LEVELDB_INCLUDE_DIRS})
include_directories(..)

set(EXECUTABLE web3jsonrpc)

file(GLOB HEADERS "*.h")

if (ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})
endif()

target_link_libraries(${EXECUTABLE} ${LEVELDB_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${JSONCPP_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_SERVER_LIBRARIES})

target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} secp256k1)
target_link_libraries(${EXECUTABLE} solidity)

if (NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
	target_link_libraries(${EXECUTABLE} serpent)
endif()

if (ETH_JSON_RPC_STUB)
	add_custom_target(jsonrpcstub) 
	add_custom_command(TARGET jsonrpcstub
		POST_BUILD
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMAND ${CMAKE_COMMAND} -DETH_SPEC_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spec.json" -DETH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
			-DETH_SERVER_DIR="${CMAKE_CURRENT_SOURCE_DIR}" -DETH_CLIENT_DIR="${CMAKE_SOURCE_DIR}/test"
			-DETH_SERVER_NAME=AbstractWebThreeStubServer -DETH_CLIENT_NAME=WebThreeStubClient
			-DETH_JSON_RPC_STUB="${ETH_JSON_RPC_STUB}"
			-P "${ETH_SCRIPTS_DIR}/jsonrpcstub.cmake"
	)
	add_dependencies(${EXECUTABLE} jsonrpcstub)
endif()

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )


