if(APPLE)
	set(CONFIG_CMD  ./configure --with-darwinssl --prefix=${ETH_DEPENDENCY_INSTALL_DIR} --exec-prefix=${ETH_DEPENDENCY_INSTALL_DIR})
else()
    set (CONFIG_CMD  ./configure --prefix=${ETH_DEPENDENCY_INSTALL_DIR} --exec-prefix=${ETH_DEPENDENCY_INSTALL_DIR})
endif()

ExternalProject_Add(curl
	URL http://curl.haxx.se/download/curl-7.38.0.tar.bz2 
	BINARY_DIR curl-prefix/src/curl
	CONFIGURE_COMMAND ${CONFIG_CMD}
	BUILD_COMMAND make -j 3
	INSTALL_COMMAND make install
)
