if (APPLE)
	ExternalProject_add(qt
		URL http://qtmirror.ics.com/pub/qtproject/official_releases/qt/5.3/5.3.2/single/qt-everywhere-opensource-src-5.3.2.tar.gz
		BINARY_DIR qt-prefix/src/qt
		PATCH_COMMAND patch -d qtmultimedia/src/plugins/avfoundation/mediaplayer < ${CMAKE_CURRENT_SOURCE_DIR}/compile/qt_osx.patch
		CONFIGURE_COMMAND ./configure -prefix ${ETH_DEPENDENCY_INSTALL_DIR} -system-zlib -qt-libpng -qt-libjpeg -confirm-license -opensource -nomake tests -release -nomake examples -no-xcb -arch x86_64
		BUILD_COMMAND make
		INSTALL_COMMAND make install
	)
elseif(WIN32)
	ExternalProject_Add(qt
		DEPENDS icu jom
		URL http://qtmirror.ics.com/pub/qtproject/official_releases/qt/5.3/5.3.2/single/qt-everywhere-opensource-src-5.3.2.tar.gz
		BINARY_DIR qt-prefix/src/qt
		UPDATE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/compile/qt_tools.bat
		#PATCH_COMMAND cmake -E copy ${CMAKE_CURRENT_SOURCE_DIR}/compile/qt_configure.bat qtbase/configure.bat
		CONFIGURE_COMMAND configure -prefix ${ETH_DEPENDENCY_INSTALL_DIR} -opensource -confirm-license -release -opengl desktop -platform win32-msvc2013 -icu -I ${ETH_DEPENDENCY_INSTALL_DIR}/include -L ${ETH_DEPENDENCY_INSTALL_DIR}/lib -nomake tests -nomake examples
		BUILD_COMMAND nmake
		INSTALL_COMMAND nmake install
	)

	ExternalProject_Add_Step(qt configure_paths
		COMMAND set PATH=${ETH_DEPENDENCY_INSTALL_DIR}/bin;%cd%/gnuwin32/bin;%cd%/qtbase/bin;%PATH%
		DEPENDEES patch
		DEPENDERS configure
	)

else()

endif()


