if(APPLE)

elseif(WIN32)
ExternalProject_Add(qt
	DEPENDS icu jom
	URL http://download.qt-project.org/official_releases/qt/5.2/5.2.1/single/qt-everywhere-opensource-src-5.2.1.tar.gz
	BINARY_DIR qt-prefix/src/qt
	UPDATE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools.bat
	PATCH_COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/configure.bat qtbase
	CONFIGURE_COMMAND configure -prefix ${ETH_DEPENDENCY_INSTALL_DIR} -opensource -confirm-license -release -opengl desktop -platform win32-msvc2013 -icu -I ${ETH_DEPENDENCY_INSTALL_DIR}/include -L ${ETH_DEPENDENCY_INSTALL_DIR}/lib -nomake tests -nomake examples

	BUILD_COMMAND nmake
	INSTALL_COMMAND nmake install
	)

ExternalProject_Add_Step(qt configure_paths
	COMMAND set PATH=${ETH_DEPENDENCY_INSTALL_DIR}/bin;%cd%/gnuwin32/bin;%cd%/qtbase/bin;%PATH%
	DEPENDEES patch
	DEPENDERS configure
	)

#ExternalProject_Add_Step(qt configure_visual
#	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools.bat
#	DEPENDEES patch
#	DEPENDERS configure
#	)

else()

endif()


