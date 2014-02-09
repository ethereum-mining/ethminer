cmake_policy(SET CMP0015 NEW)

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

if(APPLE AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
	set(CMAKE_INSTALL_PREFIX "./")
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]")
	add_definitions(-DQT_NO_DEBUG_OUTPUT)
	set(CMAKE_BUILD_RELEASE TRUE)
else()
	set(CMAKE_BUILD_RELEASE FALSE)
endif()


# set(ICONS_DIR "${${PROJECT_NAME}_SOURCE_DIR}/icons")


if(!APPLE)
	add_executable(alethzero Main.ui ${SRC_LIST})
	qt5_wrap_ui(ui_Main.h Main.ui)
	qt5_use_modules(alethzero Widgets Network )
	target_link_libraries(alethzero ethereum)
	install( TARGETS alethzero RUNTIME DESTINATION bin )
else()
	set(BIN_INSTALL_DIR ".")
	set(DOC_INSTALL_DIR ".")

	set(PROJECT_VERSION "${ETH_VERSION}")
	set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_COPYRIGHT "${PROJECT_COPYRIGHT_YEAR} ${PROJECT_VENDOR}")
	# set(MACOSX_BUNDLE_ICON_FILE "audio-input-microphone.icns")
	set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")
	set(MACOSX_BUNDLE_BUNDLE_NAME "AlethZero")

	# set(MACOSX_BUNDLE_RESOURCES "${CMAKE_CURRENT_BINARY_DIR}/${MACOSX_BUNDLE_BUNDLE_NAME}.app/Contents/Resources")
	# set(MACOSX_BUNDLE_ICON "${ICONS_DIR}/${MACOSX_BUNDLE_ICON_FILE}")
	# execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${MACOSX_BUNDLE_RESOURCES})
	# execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MACOSX_BUNDLE_ICON} ${MACOSX_BUNDLE_RESOURCES})
# 	SET(CPACK_GENERATOR "DragNDrop")
# 	SET(CPACK_DMG_FORMAT "UDBZ")
# 	SET(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
# 	SET(CPACK_SYSTEM_NAME "OSX")
# 	SET(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}")
# # set(CPACK_PACKAGE_ICON "${ICONS_DIR}/DMG.icns")
# # set(CPACK_DMG_DS_STORE "${ICONS_DIR}/DMGDSStore")
# # set(CPACK_DMG_BACKGROUND_IMAGE "${ICONS_DIR}/DMGBackground.png")
# 	INCLUDE(CPack)
	
	set(EXECUTABLE "AlethZero.app")
	INCLUDE(BundleUtilities)
	
	ADD_EXECUTABLE(
		AlethZero
		MACOSX_BUNDLE
		Main.ui ${SRC_LIST}
	)
	
	set_target_properties(AlethZero PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}//EthereumMacOSXBundleInfo.plist.in")

	qt5_wrap_ui(ui_Main.h Main.ui)
	qt5_use_modules(AlethZero Widgets Network)
	target_link_libraries(AlethZero ethereum)

	SET_SOURCE_FILES_PROPERTIES(
		AlethZero
		PROPERTIES
		MACOSX_PACKAGE_LOCATION MacOS
	)
	
	# install( TARGETS AlethZero RUNTIME DESTINATION bin )
	
	# fixup_bundle("${EXECUTABLE}" "" "${MACOSX_BUNDLE_RESOURCES}")

	# set(APPS AlethZero)
	# install(CODE "
	# 	include(BundleUtilities)
	# 	set(BU_CHMOD_BUNDLE_ITEMS 1)
	# 	fixup_bundle(\"${APPS}\" \"${BUNDLELIBS}\" \"${SEARCHDIRS}\")
	# 	" COMPONENT RUNTIME )
endif()




# if(APPLE)
# 	set(MY_APP_NAME AlethZero)
# 	set(MY_APP_VERSION ETH_VERSION)
# 	set(MACOSX_BUNDLE true)
# 	set(MACOSX_BUNDLE_BUNDLE_NAME ${MY_APP_NAME})
# 	set(MACOSX_BUNDLE_INFO_STRING "${MY_APP_NAME} ${MY_APP_VERSION}")
# 	set(MACOSX_BUNDLE_LONG_VERSION_STRING "${MY_APP_VERSION}")
# 	set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${MY_APP_VERSION}")
# 	set(MACOSX_BUNDLE_BUNDLE_VERSION "${MY_APP_VERSION}")
# # set(MACOSX_BUNDLE_ICON_FILE "${PROJECT_SOURCE_DIR}/${MY_APP_NAME}.icns")
# 
# # ADD_EXECUTABLE(
# # 	alerthzero
# # 	MACOSX_BUNDLE
# # )
# 
# 	SET_SOURCE_FILES_PROPERTIES(
# 		alethzero
# 		PROPERTIES
# 		MACOSX_PACKAGE_LOCATION MacOS
# 	)
# 	# SET_SOURCE_FILES_PROPERTIES(
# 	# 	PROPERTIES
# 	# 	MACOSX_PACKAGE_LOCATION Resources
# 	# )
# endif()