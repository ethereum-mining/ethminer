if (APPLE)
	ExternalProject_Add(snappy
		URL https://snappy.googlecode.com/files/snappy-1.1.1.tar.gz
		BINARY_DIR snappy-prefix/src/snappy
		CONFIGURE_COMMAND ./configure --disable-dependency-tracking --prefix=${ETH_DEPENDENCY_INSTALL_DIR} 
		BUILD_COMMAND ""
		INSTALL_COMMAND make install
	)
elseif(WIN32)

else()

endif()

