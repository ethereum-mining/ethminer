include(ExternalProject)

ExternalProject_Add(curl
	URL http://curl.haxx.se/download/curl-7.38.0.tar.bz2 
	BINARY_DIR curl-prefix/src/curl
	CONFIGURE_COMMAND ./configure --with-darwinssl
	BUILD_COMMAND make -j 3
	INSTALL_COMMAND ""
	)


