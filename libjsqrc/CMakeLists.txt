cmake_policy(SET CMP0015 NEW)
# let cmake autolink dependencies on windows
cmake_policy(SET CMP0020 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	cmake_policy(SET CMP0043 OLD)
endif()
set(CMAKE_AUTOMOC OFF)

qt5_add_resources(JSQRC js.qrc)
add_library(jsqrc STATIC ${JSQRC})
target_link_libraries(jsqrc Qt5::Core)

install( TARGETS jsqrc RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )

file(GLOB_RECURSE JSFILES "ethereumjs/lib/*.js")
add_custom_target(aux_js SOURCES ${JSFILES})
