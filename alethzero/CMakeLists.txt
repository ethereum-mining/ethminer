cmake_policy(SET CMP0015 NEW)

set(CMAKE_AUTOMOC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(..)
link_directories(../libethereum)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	execute_process(
		COMMAND brew info qt5 | grep Cellar | awk -F ' ' '{print $1}'
		OUTPUT_VARIABLE osx_qt5
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	set(CMAKE_PREFIX_PATH /usr/local/Cellar/qt5/5.2.0)
	include_directories(/usr/local/Cellar/qt5/5.2.0/include /usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

find_package(Qt5Widgets REQUIRED)

add_executable(alethzero Main.ui ${SRC_LIST})
qt5_wrap_ui(ui_Main.h Main.ui)

qt5_use_modules(alethzero Widgets Network )
target_link_libraries(alethzero ethereum)

install( TARGETS alethzero RUNTIME DESTINATION bin )
