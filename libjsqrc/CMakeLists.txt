cmake_policy(SET CMP0015 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (APPLE)
	cmake_policy(SET CMP0043 OLD)
endif()
set(CMAKE_AUTOMOC OFF)

qt5_add_resources(JSQRC js.qrc)
add_library(jsqrc STATIC ${JSQRC})
qt5_use_modules(jsqrc Core)

install( TARGETS jsqrc ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
