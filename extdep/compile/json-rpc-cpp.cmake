# json-rpc-cpp is under heavy development, not yet stable, and multiplatform builds are not yet available. 
# DO NOT MESS WITH THESE SETTINGS! IF YOU HAVE TO MAKE CHANGES HERE, CONSULT sven@ethdev.com BEFOREHAND!!

# DO NOT CHANGE ANYTHING HERE!
if(APPLE)
	ExternalProject_Add(json-rpc-cpp
		# DEPENDS argtable2 jsoncpp
		# DEPENDS curl # re-enable later, when we build curl again
		GIT_REPOSITORY https://github.com/cinemast/libjson-rpc-cpp.git 
		GIT_TAG v0.3.2
		BINARY_DIR json-rpc-cpp-prefix/src/json-rpc-cpp
		CONFIGURE_COMMAND cmake -DCMAKE_INSTALL_PREFIX=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_MODULE_PATH:PATH=${CMAKE_CURRENT_SOURCE_DIR}/cmake -DETH_DEPENDENCY_INSTALL_DIR:PATH=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_BUILD_TYPE=None -DCMAKE_FIND_FRAMEWORK=LAST -Wno-dev .
		BUILD_COMMAND make -j 3
		INSTALL_COMMAND make install && ${CMAKE_CURRENT_SOURCE_DIR}/scripts/json-rpc-cpp_osx.sh . ${ETH_DEPENDENCY_INSTALL_DIR} 
	)

elseif (WIN32)
	ExternalProject_Add(json-rpc-cpp
		DEPENDS argtable2 jsoncpp curl
		GIT_REPOSITORY https://github.com/debris/libjson-rpc-cpp.git 
		GIT_TAG windows
		BINARY_DIR json-rpc-cpp-prefix/src/json-rpc-cpp
		CONFIGURE_COMMAND cmake -DCMAKE_PREFIX_PATH=${ETH_DEPENDENCY_INSTALL_DIR} -DCURL_LIBRARIES=${ETH_DEPENDENCY_INSTALL_DIR}/lib/libcurl.lib .
		BUILD_COMMAND devenv libjson-rpc-cpp.sln /build release
		INSTALL_COMMAND cmd /c cp lib/Release/* ${ETH_DEPENDENCY_INSTALL_DIR}/lib && cp -R src/jsonrpccpp ${ETH_DEPENDENCY_INSTALL_DIR}/include
	)
else()
	ExternalProject_Add(json-rpc-cpp
		# DEPENDS argtable2 jsoncpp
		# DEPENDS curl # re-enable later, when we build curl again
		GIT_REPOSITORY https://github.com/cinemast/libjson-rpc-cpp.git 
		GIT_TAG v0.3.2
		BINARY_DIR json-rpc-cpp-prefix/src/json-rpc-cpp
		CONFIGURE_COMMAND cmake -DCMAKE_INSTALL_PREFIX=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_MODULE_PATH:PATH=${CMAKE_CURRENT_SOURCE_DIR}/cmake -DETH_DEPENDENCY_INSTALL_DIR:PATH=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_BUILD_TYPE=None -DCMAKE_FIND_FRAMEWORK=LAST .
		BUILD_COMMAND make -j 3
		INSTALL_COMMAND make install
	)

endif()

