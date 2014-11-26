if(APPLE)
ExternalProject_Add(leveldb
    DEPENDS snappy
	URL https://leveldb.googlecode.com/files/leveldb-1.15.0.tar.gz 
	BINARY_DIR leveldb-prefix/src/leveldb
	CONFIGURE_COMMAND patch < ${CMAKE_CURRENT_SOURCE_DIR}/leveldb_osx.patch
	BUILD_COMMAND export ETH_DEPENDENCY_INSTALL_DIR=${ETH_DEPENDENCY_INSTALL_DIR} && make -j 3
	INSTALL_COMMAND cp -rf include/leveldb ${ETH_DEPENDENCY_INSTALL_DIR}/include/ && mv libleveldb.a ${ETH_DEPENDENCY_INSTALL_DIR}/lib && mv libleveldb.dylib.1.15 ${ETH_DEPENDENCY_INSTALL_DIR}/lib/libleveldb.dylib
	)
elseif(WIN32)
ExternalProject_Add(leveldb
	GIT_REPOSITORY https://code.google.com/p/leveldb
	GIT_TAG windows
	BINARY_DIR leveldb-prefix/src/leveldb
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	)
else()

endif()

