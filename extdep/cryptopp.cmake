if(${APPLE})
ExternalProject_Add(cryptopp
	URL http://www.cryptopp.com/cryptopp562.zip
	BINARY_DIR cryptopp-prefix/src/cryptopp
	CONFIGURE_COMMAND ""
	BUILD_COMMAND make -j 3
	INSTALL_COMMAND make dynamic install PREFIX=${ETH_DEPENDENCY_INSTALL_DIR}
	)
else()
ExternalProject_Add(cryptopp
	URL https://github.com/mmoss/cryptopp/archive/v5.6.2.zip
	BINARY_DIR cryptopp-prefix/src/cryptopp
	CONFIGURE_COMMAND ""
	BUILD_COMMAND scons --shared --prefix=${ETH_DEPENDENCY_INSTALL_DIR}
	INSTALL_COMMAND ""
)
endif()
	
