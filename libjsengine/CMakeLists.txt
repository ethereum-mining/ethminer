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

set(EXECUTABLE jsengine)

file(GLOB HEADERS "*.h")

include(EthUtils)
eth_add_resources("${CMAKE_CURRENT_SOURCE_DIR}/JSResources.cmake" "JSRES")
message(STATUS "HERE!!! ${JSRES}")
add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS} ${JSRES})

add_dependencies(${EXECUTABLE} BuildInfo.h)

# macos brew version of v8 needs to be compiled with libstdc++
# it also needs to be dynamic library
# xcode needs libstdc++ to be explicitly set as it's attribute
if (APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
	set_property(TARGET ${EXECUTABLE} PROPERTY XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libstdc++")
endif()

target_link_libraries(${EXECUTABLE} ${V8_LIBRARIES})

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
