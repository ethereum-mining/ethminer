if(${APPLE})
ExternalProject_Add(curl
	URL http://curl.haxx.se/download/curl-7.38.0.tar.bz2 
	BINARY_DIR curl-prefix/src/curl
	CONFIGURE_COMMAND ./configure --with-darwinssl --prefix=${ETH_DEPENDENCY_INSTALL_DIR} --exec-prefix=${ETH_DEPENDENCY_INSTALL_DIR}
	BUILD_COMMAND make -j 3
	INSTALL_COMMAND make install
	)
else()

endif()

