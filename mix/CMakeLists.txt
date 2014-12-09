set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)
include_directories(..)

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

target_link_libraries(${EXECUTEABLE} Qt5::Core)
target_link_libraries(${EXECUTEABLE} Qt5::Gui)
target_link_libraries(${EXECUTEABLE} webthree)
target_link_libraries(${EXECUTEABLE} qethereum)
target_link_libraries(${EXECUTEABLE} ethereum)
target_link_libraries(${EXECUTEABLE} evm)
target_link_libraries(${EXECUTEABLE} ethcore)
target_link_libraries(${EXECUTEABLE} devcrypto)
target_link_libraries(${EXECUTEABLE} secp256k1)
target_link_libraries(${EXECUTEABLE} serpent)
target_link_libraries(${EXECUTEABLE} lll)
target_link_libraries(${EXECUTEABLE} solidity)
target_link_libraries(${EXECUTEABLE} evmcore)
target_link_libraries(${EXECUTEABLE} devcore)
target_link_libraries(${EXECUTEABLE} web3jsonrpc)
target_link_libraries(${EXECUTEABLE} jsqrc)

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
else()
	install( TARGETS ${EXECUTEABLE} RUNTIME DESTINATION bin)
endif ()
