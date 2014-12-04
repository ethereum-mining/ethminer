set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)
include_directories(..)


if (APPLE)
	# Add homebrew path for qt5
	set(CMAKE_PREFIX_PATH /usr/local/opt/qt5)
	include_directories(/usr/local/opt/qt5/include /usr/local/include)
elseif ("${TARGET_PLATFORM}" STREQUAL "w64")
	set(SRC_LIST ${SRC_LIST} ../windows/qt_plugin_import.cpp)
	include_directories(/usr/x86_64-w64-mingw32/include /usr/x86_64-w64-mingw32/include/QtCore /usr/x86_64-w64-mingw32/include/QtGui /usr/x86_64-w64-mingw32/include/QtQuick /usr/x86_64-w64-mingw32/include/QtQml /usr/x86_64-w64-mingw32/include/QtNetwork /usr/x86_64-w64-mingw32/include/QtWidgets /usr/x86_64-w64-mingw32/include/QtWebKit /usr/x86_64-w64-mingw32/include/QtWebKitWidgets)
elseif (UNIX)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ";$ENV{QTDIR}/lib/cmake")
endif ()

find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Quick REQUIRED)
find_package(Qt5Qml REQUIRED)
find_package(Qt5Network REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5WebKit REQUIRED)
find_package(Qt5WebKitWidgets REQUIRED)

#qt5_wrap_ui(ui_Main.h Main.ui)

qt5_add_resources(UI_RESOURCES qml.qrc)

# Set name of binary and add_executable()
file(GLOB HEADERS "*.h")
if (APPLE)
	set(EXECUTEABLE mix)
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
	set(MACOSX_BUNDLE_ICON_FILE mix)
	include(BundleUtilities)

		add_executable(${EXECUTEABLE} MACOSX_BUNDLE ${SRC_LIST} ${HEADERS} ${UI_RESOURCES})
	set_target_properties(${EXECUTEABLE} PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/EthereumMacOSXBundleInfo.plist.in")
	SET_SOURCE_FILES_PROPERTIES(${EXECUTEABLE} PROPERTIES MACOSX_PACKAGE_LOCATION MacOS)
	SET_SOURCE_FILES_PROPERTIES(${MACOSX_BUNDLE_ICON_FILE}.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

else ()
	set(EXECUTEABLE mix)
		add_executable(${EXECUTEABLE} ${SRC_LIST} ${HEADERS} ${UI_RESOURCES})
endif ()

qt5_use_modules(${EXECUTEABLE} Core)# Gui Widgets Network WebKit WebKitWidgets)
target_link_libraries(${EXECUTEABLE} webthree qethereum ethereum evm ethcore devcrypto secp256k1 gmp ${CRYPTOPP_LS} serpent lll solidity evmcore devcore web3jsonrpc jsqrc)

if (APPLE)
	# First have qt5 install plugins and frameworks
	add_custom_command(TARGET ${EXECUTEABLE} POST_BUILD
		COMMAND /usr/local/opt/qt5/bin/macdeployqt -qmldir=${CMAKE_CURRENT_SOURCE_DIR}/qml ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${EXECUTEABLE}.app
		WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

	# This tool and next will inspect linked libraries in order to determine which dependencies are required
	if (${CMAKE_CFG_INTDIR} STREQUAL ".")
		set(APP_BUNDLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${EXECUTEABLE}.app")
	else ()
		set(APP_BUNDLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/\$ENV{CONFIGURATION}/${EXECUTEABLE}.app")
	endif ()
	install(CODE "
		include(BundleUtilities)
		set(BU_CHMOD_BUNDLE_ITEMS 1)
		fixup_bundle(\"${APP_BUNDLE_PATH}\" \"${BUNDLELIBS}\" \"../libqethereum ../libethereum ../secp256k1\")
		" COMPONENT RUNTIME )
	# Cleanup duplicate libs from macdeployqt
	install(CODE "
		file(GLOB LINGER_RM \"${APP_BUNDLE_PATH}/Contents/Frameworks/*.dylib\")
		if (LINGER_RM)
			file(REMOVE \${LINGER_RM})
		endif ()
	")
elseif (UNIX)
else ()
	target_link_libraries(${EXECUTEABLE} boost_system)
	target_link_libraries(${EXECUTEABLE} boost_filesystem)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTEABLE} ${CMAKE_THREAD_LIBS_INIT})
	install( TARGETS ${EXECUTEABLE} RUNTIME DESTINATION bin )
endif ()

qt5_use_modules(${EXECUTEABLE} Core Gui)
