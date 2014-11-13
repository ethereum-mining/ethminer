if(${APPLE})
ExternalProject_Add(cryptopp
     URL https://downloads.sourceforge.net/project/cryptopp/cryptopp/5.6.2/cryptopp562.zip
     BINARY_DIR cryptopp-prefix/src/cryptopp
     CONFIGURE_COMMAND ""
     BUILD_COMMAND make CXX=clang++ CXXFLAGS=-DCRYPTOPP_DISABLE_ASM
     INSTALL_COMMAND make install PREFIX=${ETH_DEPENDENCY_INSTALL_DIR}
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
	
