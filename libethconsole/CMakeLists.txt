cmake_policy(SET CMP0015 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	# old policy do not use MACOSX_RPATH
	cmake_policy(SET CMP0042 OLD)
endif()

set(CMAKE_AUTOMOC OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${V8_INCLUDE_DIRS})

set(EXECUTABLE ethconsole)

file(GLOB HEADERS "*.h")

add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})

target_link_libraries(${EXECUTABLE} ${V8_LIBRARIES})

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

