if (APPLE)
	ExternalProject_Add(curl
		URL http://curl.haxx.se/download/curl-7.38.0.tar.bz2 
		BINARY_DIR curl-prefix/src/curl
		CONFIGURE_COMMAND ./configure --with-darwinssl --prefix=${ETH_DEPENDENCY_INSTALL_DIR} --exec-prefix=${ETH_DEPENDENCY_INSTALL_DIR}
		BUILD_COMMAND make -j 3
		INSTALL_COMMAND make install
	)
elseif (WIN32)
	ExternalProject_Add(curl
		GIT_REPOSITORY https://github.com/debris/libcurl-7.29
		GIT_TAG master
		BINARY_DIR curl-prefix/src/curl
		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND cmd /c cp lib/release/libcurl.lib ${ETH_DEPENDENCY_INSTALL_DIR}/lib && cp -R include/curl ${ETH_DEPENDENCY_INSTALL_DIR}/include
	)

else()
	ExternalProject_Add(curl
		URL http://curl.haxx.se/download/curl-7.38.0.tar.bz2 
		BINARY_DIR curl-prefix/src/curl
		CONFIGURE_COMMAND CONFIG_CMD  ./configure --prefix=${ETH_DEPENDENCY_INSTALL_DIR} --exec-prefix=${ETH_DEPENDENCY_INSTALL_DIR}
		BUILD_COMMAND make -j 3
		INSTALL_COMMAND make install
	)

endif()

