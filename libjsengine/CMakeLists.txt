cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ${V8_INCLUDE_DIRS})
include_directories(BEFORE ..)

set(EXECUTABLE jsengine)

file(GLOB HEADERS "*.h")

include(EthUtils)
eth_add_resources("${CMAKE_CURRENT_SOURCE_DIR}/JSResources.cmake" "JSRES")
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
