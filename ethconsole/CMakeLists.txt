cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})
include_directories(${V8_INCLUDE_DIRS})

set(EXECUTABLE ethconsole)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
target_link_libraries(${EXECUTABLE} ${Boost_REGEX_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${CURL_LIBRARIES})

if (DEFINED WIN32 AND NOT DEFINED CMAKE_COMPILER_IS_MINGW)
	eth_copy_dlls(${EXECUTABLE} CURL_DLLS)
endif()
target_link_libraries(${EXECUTABLE} jsconsole)
target_link_libraries(${EXECUTABLE} devcore)

if (APPLE)
	install(TARGETS ${EXECUTABLE} DESTINATION bin)
else()
	eth_install_executable(${EXECUTABLE})
endif()

