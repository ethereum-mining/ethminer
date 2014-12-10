cmake_policy(SET CMP0015 NEW)
# let cmake autolink dependencies on windows
cmake_policy(SET CMP0020 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	cmake_policy(SET CMP0043 OLD)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)
include_directories(..)

qt5_add_resources(UI_RESOURCES qml.qrc)

# Set name of binary and add_executable()
file(GLOB HEADERS "*.h")
if (APPLE)
	set(EXECUTABLE mix)
	set(BIN_INSTALL_DIR ".")
	set(DOC_INSTALL_DIR ".")

	set(PROJECT_VERSION "${ETH_VERSION}")
	set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_COPYRIGHT "${PROJECT_COPYRIGHT_YEAR} ${PROJECT_VENDOR}")
	set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")
	set(MACOSX_BUNDLE_BUNDLE_NAME ${EXECUTABLE})
	set(MACOSX_BUNDLE_ICON_FILE mix)
	include(BundleUtilities)

		add_executable(${EXECUTABLE} MACOSX_BUNDLE ${SRC_LIST} ${HEADERS} ${UI_RESOURCES})
	set_target_properties(${EXECUTABLE} PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/EthereumMacOSXBundleInfo.plist.in")
	SET_SOURCE_FILES_PROPERTIES(${EXECUTABLE} PROPERTIES MACOSX_PACKAGE_LOCATION MacOS)
	SET_SOURCE_FILES_PROPERTIES(${MACOSX_BUNDLE_ICON_FILE}.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

else ()
	set(EXECUTABLE mix)
	add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS} ${UI_RESOURCES})
endif ()

target_link_libraries(${EXECUTABLE} Qt5::Core)
target_link_libraries(${EXECUTABLE} Qt5::Gui)
target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} qethereum)
target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} evm)
target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} secp256k1)
target_link_libraries(${EXECUTABLE} serpent)
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} solidity)
target_link_libraries(${EXECUTABLE} evmcore)
target_link_libraries(${EXECUTABLE} devcore)
target_link_libraries(${EXECUTABLE} web3jsonrpc)
target_link_libraries(${EXECUTABLE} jsqrc)

if (APPLE)
	# First have qt5 install plugins and frameworks
	add_custom_command(TARGET ${EXECUTABLE} POST_BUILD
		COMMAND ${CMAKE_DEPENDENCY_INSTALL_DIR}/bin/macdeployqt -qmldir=${CMAKE_CURRENT_SOURCE_DIR}/qml ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${EXECUTABLE}.app
		WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

	# This tool and next will inspect linked libraries in order to determine which dependencies are required
	if (${CMAKE_CFG_INTDIR} STREQUAL ".")
		set(APP_BUNDLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}.app")
	else ()
		set(APP_BUNDLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/\$ENV{CONFIGURATION}/${EXECUTABLE}.app")
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
	install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin)
endif ()

