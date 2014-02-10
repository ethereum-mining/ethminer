cmake_policy(SET CMP0015 NEW)

if ("${TARGET_PLATFORM}" STREQUAL "w64")
	cmake_policy(SET CMP0020 NEW)
endif ()

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

if (${TARGET_PLATFORM} STREQUAL "w64")
	set(SRC_LIST ${SRC_LIST} ../windows/qt_plugin_import.cpp)
else ()
endif ()

find_package(Qt5Widgets REQUIRED)

add_executable(alethzero Main.ui ${SRC_LIST})
qt5_wrap_ui(ui_Main.h Main.ui)

target_link_libraries(alethzero ethereum)

if (${TARGET_PLATFORM} STREQUAL "w64")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-keep-inline-dllexport -static-libgcc -static-libstdc++ -static")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-s -Wl,-subsystem,windows -mthreads -L/usr/x86_64-w64-mingw32/plugins/platforms")
	target_link_libraries(alethzero gcc)
	target_link_libraries(alethzero mingw32 qtmain mswsock iphlpapi qwindows shlwapi Qt5PlatformSupport gdi32 comdlg32 oleaut32 imm32 winmm ole32 uuid ws2_32)
	target_link_libraries(alethzero boost_system-mt-s)
	target_link_libraries(alethzero boost_filesystem-mt-s)
	target_link_libraries(alethzero boost_thread_win32-mt-s)
	target_link_libraries(alethzero Qt5PlatformSupport)
	set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
else ()
	target_link_libraries(alethzero boost_system)
	target_link_libraries(alethzero boost_filesystem)
	find_package(Threads REQUIRED)
	target_link_libraries(alethzero ${CMAKE_THREAD_LIBS_INIT})
endif ()

qt5_use_modules(alethzero Core Gui Widgets Network)

install( TARGETS alethzero RUNTIME DESTINATION bin )
