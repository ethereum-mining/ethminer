# CryptoPP does not have good cross-platform support, there exist several different other projects to make it work ... 

# TODO the OS X build throws a lot of warnings, but compiles fine
if (APPLE)
	ExternalProject_Add(cryptopp
		URL https://downloads.sourceforge.net/project/cryptopp/cryptopp/5.6.2/cryptopp562.zip
		BINARY_DIR cryptopp-prefix/src/cryptopp
		CONFIGURE_COMMAND ""
		BUILD_COMMAND make CXX=clang++ CXXFLAGS=-DCRYPTOPP_DISABLE_ASM
		INSTALL_COMMAND make install PREFIX=${ETH_DEPENDENCY_INSTALL_DIR}
	)
elseif (WIN32)
	file(MAKE_DIRECTORY ${ETH_DEPENDENCY_INSTALL_DIR}/include/cryptopp)

	ExternalProject_Add(cryptopp
		SVN_REPOSITORY http://svn.code.sf.net/p/cryptopp/code/trunk/c5
		SVN_REVISION -r "541"
		BINARY_DIR cryptopp-prefix/src/cryptopp
		CONFIGURE_COMMAND devenv cryptest.sln /upgrade
		BUILD_COMMAND devenv cryptest.sln /build release
		INSTALL_COMMAND cmd /c cp Win32/DLL_Output/Release/cryptopp.dll ${ETH_DEPENDENCY_INSTALL_DIR}/lib && cp Win32/DLL_Output/Release/cryptopp.lib ${ETH_DEPENDENCY_INSTALL_DIR}/lib && cp *.h ${ETH_DEPENDENCY_INSTALL_DIR}/include/cryptopp
	)
# on Linux, the default Makefile does not work.
else()
	ExternalProject_Add(cryptopp
		URL https://github.com/mmoss/cryptopp/archive/v5.6.2.zip
		BINARY_DIR cryptopp-prefix/src/cryptopp
		CONFIGURE_COMMAND ""
		BUILD_COMMAND scons --shared --prefix=${ETH_DEPENDENCY_INSTALL_DIR}
		INSTALL_COMMAND ""
	)
endif()

