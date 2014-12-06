cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

qt5_add_resources(JSQRC js.qrc)
add_library(jsqrc STATIC ${JSQRC})
qt5_use_modules(jsqrc Core)

install( TARGETS jsqrc ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
