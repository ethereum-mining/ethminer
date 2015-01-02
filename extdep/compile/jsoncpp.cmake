if (APPLE)

elseif (WIN32)

	file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/include/jsoncpp)
	ExternalProject_Add(jsoncpp
		GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
		GIT_TAG svn-import
		BINARY_DIR jsoncpp-prefix/src/jsoncpp
		CONFIGURE_COMMAND cmake .
		BUILD_COMMAND devenv jsoncpp.sln /build release
		INSTALL_COMMAND cmd /c cp lib/Release/jsoncpp.lib ${ETH_DEPENDENCY_INSTALL_DIR}/lib && cp -R include/json ${ETH_DEPENDENCY_INSTALL_DIR}/include/jsoncpp
	)

else()
endif()
