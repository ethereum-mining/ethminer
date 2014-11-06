ExternalProject_Add(
	cryptopp
	URL https://github.com/mmoss/cryptopp/archive/v5.6.2.zip
	BINARY_DIR cryptopp-prefix/src/cryptopp
	CONFIGURE_COMMAND ""
	BUILD_COMMAND scons --shared --prefix=${ETH_DEPENDENCY_INSTALL_DIR}
	INSTALL_COMMAND ""
)


	
