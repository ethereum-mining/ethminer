cmake_policy(SET CMP0015 NEW)

# Find Qt5 for Apple and update src_list for windows
if (APPLE)
	# homebrew defaults to qt4 and installs qt5 as 'keg-only'
	# which places it into /usr/local/opt insteadof /usr/local.

	set(CMAKE_PREFIX_PATH /usr/local/opt/qt5)
	include_directories(/usr/local/opt/qt5/include /usr/local/include)
elseif ("${TARGET_PLATFORM}" STREQUAL "w64")
	set(SRC_LIST ${SRC_LIST} ../windows/qt_plugin_import.cpp)
elseif (UNIX)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ";$ENV{QTDIR}/lib/cmake")
endif ()

find_package(Qt5Core)

set(CMAKE_AUTOMOC OFF)

qt5_add_resources(JSQRC js.qrc)
add_library(jsqrc STATIC ${JSQRC})
qt5_use_modules(jsqrc Core)

install( TARGETS jsqrc ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
