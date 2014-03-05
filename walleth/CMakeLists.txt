cmake_policy(SET CMP0015 NEW)

if ("${TARGET_PLATFORM}" STREQUAL "w64")
	cmake_policy(SET CMP0020 NEW)
endif ()


set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(..)
link_directories(../libethereum)

# Find Qt5 for Apple and update src_list for windows
if (APPLE)
	# homebrew defaults to qt4 and installs qt5 as 'keg-only'
	# which places it into /usr/local/opt insteadof /usr/local.

	set(CMAKE_PREFIX_PATH /usr/local/opt/qt5)
	include_directories(/usr/local/opt/qt5/include /usr/local/include)
elseif (${TARGET_PLATFORM} STREQUAL "w64")
	set(SRC_LIST ${SRC_LIST} ../windows/qt_plugin_import.cpp)
endif ()


find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Quick REQUIRED)
find_package(Qt5Qml REQUIRED)
find_package(Qt5Network REQUIRED)
qt5_wrap_ui(ui_Main.h Main.ui)
qt5_add_resources(RESOURCE_ADDED Resources.qrc)

# Set name of binary and add_executable()
if (APPLE)
	set(EXECUTEABLE Walleth)
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
	set(MACOSX_BUNDLE_BUNDLE_NAME ${EXECUTEABLE})
	include(BundleUtilities)

	add_executable(${EXECUTEABLE} MACOSX_BUNDLE Main.ui ${RESOURCE_ADDED} ${SRC_LIST})
else ()
	set(EXECUTEABLE walleth)
	add_executable(${EXECUTEABLE} Main.ui ${RESOURCE_ADDED} ${SRC_LIST})
endif ()

qt5_use_modules(${EXECUTEABLE} Core Gui Widgets Network Quick Qml)
target_link_libraries(${EXECUTEABLE} ethereum secp256k1 ${CRYPTOPP_LIBRARIES})

if (APPLE)
	set_target_properties(${EXECUTEABLE} PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/EthereumMacOSXBundleInfo.plist.in")

	SET_SOURCE_FILES_PROPERTIES(${EXECUTEABLE} PROPERTIES MACOSX_PACKAGE_LOCATION MacOS)

	# This is a workaround for when the build-type defaults to Debug, and when a multi-config generator like xcode is used, where the type
	# will not be set but defaults to release.
	set(generator_lowercase "${CMAKE_GENERATOR}")
	string(TOLOWER "${CMAKE_GENERATOR}" generator_lowercase)
	if (generator_lowercase STREQUAL "xcode")
		# TODO: Not sure how to resolve this. Possibly \${TARGET_BUILD_DIR}
		set(binary_build_dir "${CMAKE_CURRENT_BINARY_DIR}/Debug")
	else ()
		set(binary_build_dir "${CMAKE_CURRENT_BINARY_DIR}")
	endif ()

	set(APPS ${binary_build_dir}/${EXECUTEABLE}.app)

	# This tool and the next will automatically looked at the linked libraries in order to determine what dependencies are required. Thus, target_link_libaries only needs to add ethereum and secp256k1 (above)
	install(CODE "
		include(BundleUtilities)
		set(BU_CHMOD_BUNDLE_ITEMS 1)
		fixup_bundle(\"${APPS}\" \"${BUNDLELIBS}\" \"../libethereum ../secp256k1\")
		" COMPONENT RUNTIME )
	
	add_custom_target(addframeworks ALL
		COMMAND /usr/local/opt/qt5/bin/macdeployqt ${binary_build_dir}/${EXECUTEABLE}.app
		WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		DEPENDS ${PROJECT_NAME}
	)

elseif (${TARGET_PLATFORM} STREQUAL "w64")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-keep-inline-dllexport -static-libgcc -static-libstdc++ -static")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-s -Wl,-subsystem,windows -mthreads -L/usr/x86_64-w64-mingw32/plugins/platforms")
	target_link_libraries(${EXECUTEABLE} gcc)
	target_link_libraries(${EXECUTEABLE} mingw32 qtmain mswsock iphlpapi qwindows shlwapi Qt5PlatformSupport gdi32 comdlg32 oleaut32 imm32 winmm ole32 uuid ws2_32)
	target_link_libraries(${EXECUTEABLE} boost_system-mt-s)
	target_link_libraries(${EXECUTEABLE} boost_filesystem-mt-s)
	target_link_libraries(${EXECUTEABLE} boost_thread_win32-mt-s)
	target_link_libraries(${EXECUTEABLE} Qt5PlatformSupport)
	set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
else ()
	target_link_libraries(${EXECUTEABLE} boost_system)
	target_link_libraries(${EXECUTEABLE} boost_filesystem)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTEABLE} ${CMAKE_THREAD_LIBS_INIT})
	install( TARGETS ${EXECUTEABLE} RUNTIME DESTINATION bin )
endif ()

