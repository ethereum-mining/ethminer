cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(${DB_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE neth)

add_executable(${EXECUTABLE} ${SRC_LIST})

add_dependencies(${EXECUTABLE} BuildInfo.h)

if (JSONRPC)
	target_link_libraries(${EXECUTABLE} web3jsonrpc)
endif()

target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} secp256k1)
target_link_libraries(${EXECUTABLE} ncurses)
target_link_libraries(${EXECUTABLE} form)

install( TARGETS ${EXECUTABLE} DESTINATION bin )

