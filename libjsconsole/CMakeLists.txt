cmake_policy(SET CMP0015 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	# old policy do not use MACOSX_RPATH
	cmake_policy(SET CMP0042 OLD)
endif()

set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ${V8_INCLUDE_DIRS})
include_directories(BEFORE ..)
if (READLINE_FOUND)
	include_directories(${READLINE_INCLUDE_DIRS})
endif()
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})

set(EXECUTABLE jsconsole)

file(GLOB HEADERS "*.h")

add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

target_link_libraries(${EXECUTABLE} jsengine)
target_link_libraries(${EXECUTABLE} devcore)
if (READLINE_FOUND)
	target_link_libraries(${EXECUTABLE} ${READLINE_LIBRARIES})
endif()
target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_SERVER_LIBRARIES})

target_link_libraries(${EXECUTABLE} ${CURL_LIBRARIES})
if (DEFINED WIN32 AND NOT DEFINED CMAKE_COMPILER_IS_MINGW)
	eth_copy_dlls(${EXECUTABLE} CURL_DLLS)
endif()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
