include(ExternalProject)

ExternalProject_Add(cryptopp
	URL http://www.cryptopp.com/cryptopp562.zip
	BINARY_DIR cryptopp-prefix/src/cryptopp
	CONFIGURE_COMMAND ""
	BUILD_COMMAND make -j 3
	INSTALL_COMMAND ""
	)


