#
# this function requires the following variables to be specified:
# ETH_VERSION
# PROJECT_NAME
# PROJECT_VERSION
# PROJECT_COPYRIGHT_YEAR
# PROJECT_VENDOR
# PROJECT_DOMAIN_SECOND
# PROJECT_DOMAIN_FIRST
# SRC_LIST
# HEADERS
#
# params:
# ICON
#

macro(eth_add_executable EXECUTABLE)
	set (extra_macro_args ${ARGN})
	set (options)
	set (one_value_args ICON)
	set (multi_value_args UI_RESOURCES)
	cmake_parse_arguments (ETH_ADD_EXECUTABLE "${options}" "${one_value_args}" "${multi_value_args}" "${extra_macro_args}")

	if (APPLE)

		add_executable(${EXECUTABLE} MACOSX_BUNDLE ${SRC_LIST} ${HEADERS} ${ETH_ADD_EXECUTABLE_UI_RESOURCES})
		set(PROJECT_VERSION "${ETH_VERSION}")
		set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
		set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_NAME} ${PROJECT_VERSION}")
		set(MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
		set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
		set(MACOSX_BUNDLE_COPYRIGHT "${PROJECT_COPYRIGHT_YEAR} ${PROJECT_VENDOR}")
		set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")
		set(MACOSX_BUNDLE_BUNDLE_NAME ${EXECUTABLE})
		set(MACOSX_BUNDLE_ICON_FILE ${ETH_ADD_EXECUTABLE_ICON})
		set_target_properties(${EXECUTABLE} PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/EthereumMacOSXBundleInfo.plist.in")
		set_source_files_properties(${EXECUTABLE} PROPERTIES MACOSX_PACKAGE_LOCATION MacOS)
		set_source_files_properties(${MACOSX_BUNDLE_ICON_FILE}.icns PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

	else ()
		add_executable(${EXECUTABLE} ${ETH_ADD_EXECUTABLE_UI_RESOURCES} ${SRC_LIST} ${HEADERS})
	endif()

endmacro()

# 
# this function requires the following variables to be specified:
# ETH_DEPENDENCY_INSTALL_DIR
#
# params: 
# QMLDIR
#

macro(eth_install_executable EXECUTABLE)

	set (extra_macro_args ${ARGN})
	set (options)
	set (one_value_args QMLDIR)
	set (multi_value_args)
	cmake_parse_arguments (ETH_INSTALL_EXECUTABLE "${options}" "${one_value_args}" "${multi_value_args}" "${extra_macro_args}")
	
	if (ETH_INSTALL_EXECUTABLE_QMLDIR)
		set(eth_qml_dir "-qmldir=${ETH_INSTALL_EXECUTABLE_QMLDIR}")
		message(STATUS "${EXECUTABLE} qmldir: ${eth_qml_dir}")
	endif()

	if (APPLE)
		# First have qt5 install plugins and frameworks
		add_custom_command(TARGET ${EXECUTABLE} POST_BUILD
			COMMAND ${MACDEPLOYQT_APP} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${EXECUTABLE}.app ${eth_qml_dir}
			WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
			
		# This tool and next will inspect linked libraries in order to determine which dependencies are required
		if (${CMAKE_CFG_INTDIR} STREQUAL ".")
			set(APP_BUNDLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}.app")
		else ()
			set(APP_BUNDLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/\$ENV{CONFIGURATION}/${EXECUTABLE}.app")
		endif ()

		# TODO check, how fixup_bundle works and if it is required
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
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

		# copy all dlls to executable directory
		# TODO improve that by copying only required dlls
		file (GLOB DLLS ${ETH_DEPENDENCY_INSTALL_DIR}/bin/*.dll)

		foreach(DLL ${DLLS})
			add_custom_command(TARGET ${EXECUTABLE} POST_BUILD
				COMMAND cmake -E copy "${DLL}" "$<TARGET_FILE_DIR:${EXECUTABLE}>"
			)
		endforeach()

		add_custom_command(TARGET ${EXECUTABLE} POST_BUILD
			COMMAND cmake -E copy_directory
			"${ETH_DEPENDENCY_INSTALL_DIR}/plugins/platforms"
			$<TARGET_FILE_DIR:${EXECUTABLE}>/platforms
		)

		install( FILES ${DLLS} 
			DESTINATION bin
			COMPONENT ${EXECUTABLE}
		)

		install( DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/plugins/platforms 
			DESTINATION bin
			COMPONENT ${EXECUTABLE}
		)

		install( TARGETS ${EXECUTABLE} RUNTIME 
			DESTINATION bin
			COMPONENT ${EXECUTABLE}
		)

	else()
		install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin)
	endif ()

endmacro()


