cmake_policy(SET CMP0015 NEW)

if ("${TARGET_PLATFORM}" STREQUAL "w64")
	cmake_policy(SET CMP0020 NEW)
endif ()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(..)
link_directories(../libethereum)

# brew install qt5 libpng 
# install X11 (now called XQuartz)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	execute_process(
		COMMAND brew info qt5 | grep Cellar | awk -F ' ' '{print $1}'
		OUTPUT_VARIABLE osx_qt5
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	set(CMAKE_PREFIX_PATH /usr/local/opt/qt5)
	include_directories(/usr/local/opt/qt5/include /usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if (${TARGET_PLATFORM} STREQUAL "w64")
	set(SRC_LIST ${SRC_LIST} ../windows/qt_plugin_import.cpp)
else ()
endif ()

find_package(Qt5Widgets REQUIRED)


if(!APPLE)
	add_executable(alethzero Main.ui ${SRC_LIST})
	qt5_wrap_ui(ui_Main.h Main.ui)
	qt5_use_modules(alethzero Widgets Network )
	target_link_libraries(alethzero ethereum)
	install( TARGETS alethzero RUNTIME DESTINATION bin )
else()
	set(CMAKE_INSTALL_PREFIX ./)
	set(BIN_INSTALL_DIR ".")
	set(DOC_INSTALL_DIR ".")

	set(PROJECT_VERSION "${ETH_VERSION}")
	set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_COPYRIGHT "${PROJECT_COPYRIGHT_YEAR} ${PROJECT_VENDOR}")
	set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")
	set(MACOSX_BUNDLE_BUNDLE_NAME "AlethZero")

	INCLUDE(BundleUtilities)

	ADD_EXECUTABLE(
		AlethZero
		MACOSX_BUNDLE
		Main.ui ${SRC_LIST}
	)
	
	set_target_properties(AlethZero PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/EthereumMacOSXBundleInfo.plist.in")

	qt5_wrap_ui(ui_Main.h Main.ui)
	qt5_use_modules(AlethZero Widgets Network)
	target_link_libraries(AlethZero ethereum)

	SET_SOURCE_FILES_PROPERTIES(
		AlethZero
		PROPERTIES
		MACOSX_PACKAGE_LOCATION MacOS
	)

	if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE Debug)
	endif()

	set(BUILD_TYPE ${CMAKE_BUILD_TYPE})
	get_filename_component(APP_INSTALL_PATH ${CMAKE_INSTALL_PREFIX} REALPATH)
	set(APPS ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/AlethZero.app) # CMAKE_CURRENT_BINARY_DIR, CMAKE_RUNTIME_OUTPUT_DIRECTORY
	install(CODE "
		include(BundleUtilities)
		set(BU_CHMOD_BUNDLE_ITEMS 1)
		fixup_bundle(\"${APPS}\" \"${BUNDLELIBS}\" \"\")
		" COMPONENT RUNTIME )
	
	add_custom_target(addframeworks ALL 	
		COMMAND /usr/local/opt/qt5/bin/macdeployqt ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/AlethZero.app
		WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		DEPENDS ${PROJECT_NAME}
	)
endif()

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
