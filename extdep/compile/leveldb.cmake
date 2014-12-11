if (APPLE)
	ExternalProject_Add(leveldb
		#DEPENDS snappy
		URL https://leveldb.googlecode.com/files/leveldb-1.15.0.tar.gz 
		BINARY_DIR leveldb-prefix/src/leveldb
		#CONFIGURE_COMMAND patch < ${CMAKE_CURRENT_SOURCE_DIR}/compile/leveldb_osx.patch
		CONFIGURE_COMMAND ""
		BUILD_COMMAND export ETH_DEPENDENCY_INSTALL_DIR=${ETH_DEPENDENCY_INSTALL_DIR} && make -j 3
		INSTALL_COMMAND cp -rf include/leveldb ${ETH_DEPENDENCY_INSTALL_DIR}/include/ && cp libleveldb.a ${ETH_DEPENDENCY_INSTALL_DIR}/lib && cp libleveldb.dylib.1.15 ${ETH_DEPENDENCY_INSTALL_DIR}/lib/libleveldb.dylib
	)
elseif (WIN32)
	ExternalProject_Add(leveldb
		GIT_REPOSITORY https://github.com/debris/leveldb-win32.git
		GIT_TAG master
		BINARY_DIR leveldb-prefix/src/leveldb
		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND cmd /c cp lib/LibLevelDB.lib ${ETH_DEPENDENCY_INSTALL_DIR}/lib/leveldb.lib && cp -R include/leveldb ${ETH_DEPENDENCY_INSTALL_DIR}/include
	)
else()

endif()

