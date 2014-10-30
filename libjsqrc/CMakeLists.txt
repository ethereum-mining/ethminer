cmake_policy(SET CMP0015 NEW)

find_package(Qt5Core)

set(CMAKE_AUTOMOC OFF)

qt5_add_resources(JSQRC js.qrc)
add_custom_target(jstargetqrc DEPENDS {JSQRC})

add_library(jsqrc STATIC ${JSQRC})
add_dependencies(jsqrc jstargetqrc)
qt5_use_modules(jsqrc Core)

install( TARGETS jsqrc ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
