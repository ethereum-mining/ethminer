ExternalProject_Add(json-rpc-cpp
	URL ${ETH_DEPENDENCY_SERVER}/json-rpc-cpp.tar.gz 
	BINARY_DIR json-rpc-cpp-prefix/src/json-rpc-cpp
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND cmake -E copy_directory . ${ETH_DEPENDENCY_INSTALL_DIR}
	)

