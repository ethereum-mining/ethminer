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
	set (multi_value_args UI_RESOURCES WIN_RESOURCES)
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
		add_executable(${EXECUTABLE} ${ETH_ADD_EXECUTABLE_UI_RESOURCES}  ${ETH_ADD_EXECUTABLE_WIN_RESOURCES} ${SRC_LIST} ${HEADERS})
	endif()

endmacro()

macro(eth_copy_dll EXECUTABLE DLL)
	# dlls must be unsubstitud list variable (without ${}) in format
	# optimized;path_to_dll.dll;debug;path_to_dlld.dll 
	list(GET ${DLL} 1 DLL_RELEASE)
	list(GET ${DLL} 3 DLL_DEBUG)
	add_custom_command(TARGET ${EXECUTABLE}
		POST_BUILD 
		COMMAND ${CMAKE_COMMAND} ARGS 
		-DDLL_RELEASE="${DLL_RELEASE}" 
		-DDLL_DEBUG="${DLL_DEBUG}" 
		-DCONF="$<CONFIGURATION>"
		-DDESTINATION="${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}" 
		-P "${ETH_SCRIPTS_DIR}/copydlls.cmake"
	)
endmacro()

macro(eth_copy_dlls EXECUTABLE)
	foreach(dll ${ARGN})
		eth_copy_dll(${EXECUTABLE} ${dll})
	endforeach(dll)
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
	set (multi_value_args DLLS)
	cmake_parse_arguments (ETH_INSTALL_EXECUTABLE "${options}" "${one_value_args}" "${multi_value_args}" "${extra_macro_args}")
	
	if (ETH_INSTALL_EXECUTABLE_QMLDIR)
		if (APPLE)
			set(eth_qml_dir "-qmldir=${ETH_INSTALL_EXECUTABLE_QMLDIR}")
		elseif (WIN32)
			set(eth_qml_dir "--qmldir ${ETH_INSTALL_EXECUTABLE_QMLDIR}")
		endif()
		message(STATUS "${EXECUTABLE} qmldir: ${eth_qml_dir}")
	endif()

	if (APPLE)
		# First have qt5 install plugins and frameworks
		add_custom_command(TARGET ${EXECUTABLE} POST_BUILD
			COMMAND ${MACDEPLOYQT_APP} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${EXECUTABLE}.app -executable=${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${EXECUTABLE}.app/Contents/MacOS/${EXECUTABLE} ${eth_qml_dir}
			WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
			COMMAND sh ${ETH_SCRIPTS_DIR}/macdeployfix.sh ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${EXECUTABLE}.app/Contents
		)
			
		# This tool and next will inspect linked libraries in order to determine which dependencies are required
		if (${CMAKE_CFG_INTDIR} STREQUAL ".")
			# TODO: This should only happen for GUI application
			set(APP_BUNDLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}.app")
		else ()
			set(APP_BUNDLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/\$ENV{CONFIGURATION}/${EXECUTABLE}.app")
		endif ()

		install(CODE "
			include(BundleUtilities)
			set(BU_CHMOD_BUNDLE_ITEMS 1)
			verify_app(\"${APP_BUNDLE_PATH}\")
			" COMPONENT RUNTIME )

	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

		get_target_property(TARGET_LIBS ${EXECUTABLE} INTERFACE_LINK_LIBRARIES)
		string(REGEX MATCH "Qt5::Core" HAVE_QT ${TARGET_LIBS})
		if ("${HAVE_QT}" STREQUAL "Qt5::Core")
			add_custom_command(TARGET ${EXECUTABLE} POST_BUILD
				COMMAND cmd /C "set PATH=${Qt5Core_DIR}/../../../bin;%PATH% && ${WINDEPLOYQT_APP} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${EXECUTABLE}.exe ${eth_qml_dir}"
				WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
			)
			#workaround for https://bugreports.qt.io/browse/QTBUG-42083
			add_custom_command(TARGET ${EXECUTABLE} POST_BUILD
				COMMAND cmd /C "(echo [Paths] & echo.Prefix=.)" > "qt.conf"
				WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR} VERBATIM
			)
		endif()

		#copy additional dlls
		foreach(dll ${ETH_INSTALL_EXECUTABLE_DLLS})
			eth_copy_dll(${EXECUTABLE} ${dll})
		endforeach(dll)

		install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Debug"
			DESTINATION .
			CONFIGURATIONS Debug
			COMPONENT ${EXECUTABLE}
		)

		install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Release"
			DESTINATION .
			CONFIGURATIONS Release
			COMPONENT ${EXECUTABLE}
		)

		install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo/"
			DESTINATION bin
			CONFIGURATIONS RelWithDebInfo
			COMPONENT ${EXECUTABLE}
		)

	else()
		install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin)
	endif ()

endmacro()


