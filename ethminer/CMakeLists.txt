cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})

set(EXECUTABLE ethminer)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

add_dependencies(${EXECUTABLE} BuildInfo.h)

target_link_libraries(${EXECUTABLE} ${Boost_REGEX_LIBRARIES})

if (READLINE_FOUND)
	target_link_libraries(${EXECUTABLE} ${READLINE_LIBRARIES})
endif()

if (JSONRPC)
	target_link_libraries(${EXECUTABLE} web3jsonrpc)
	target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_CLIENT_LIBRARIES})
	target_link_libraries(${EXECUTABLE} ${CURL_LIBRARIES})
	if (DEFINED WIN32 AND NOT DEFINED CMAKE_COMPILER_IS_MINGW)
		eth_copy_dlls(${EXECUTABLE} CURL_DLLS)
	endif()
endif()

target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} ethash)

if (DEFINED WIN32 AND NOT DEFINED CMAKE_COMPILER_IS_MINGW)
	eth_copy_dlls("${EXECUTABLE}" MHD_DLLS)
endif()

install( TARGETS ${EXECUTABLE} DESTINATION bin )

