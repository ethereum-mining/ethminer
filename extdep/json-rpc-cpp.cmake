# json-rpc-cpp is under heavy development, not yet stable, and multiplatform builds are not yet available. 
# DO NOT MESS WITH THESE SETTINGS! IF YOU HAVE TO MAKE CHANGES HERE, CONSULT sven@ethdev.com BEFOREHAND!!

set(_config_cmd cmake -DCMAKE_INSTALL_PREFIX=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_MODULE_PATH:PATH=${CMAKE_CURRENT_SOURCE_DIR} -DETH_DEPENDENCY_INSTALL_DIR:PATH=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_BUILD_TYPE=None -DCMAKE_FIND_FRAMEWORK=LAST .)


# DO NOT CHANGE ANYTHING HERE!
ExternalProject_Add(json-rpc-cpp
    # DEPENDS curl # re-enable later, when we build curl again
	GIT_REPOSITORY https://github.com/cinemast/libjson-rpc-cpp.git 
	GIT_TAG v0.3.2
	BINARY_DIR json-rpc-cpp-prefix/src/json-rpc-cpp
    CONFIGURE_COMMAND ${_config_cmd}
	BUILD_COMMAND make -j 3
	INSTALL_COMMAND make install
)
