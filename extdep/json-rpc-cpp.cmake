# json-rpc-cpp is under heavy development, and multiplatform builds are not yet available. 
# we use a forked repository which already has preliminary windows support

if(APPLE)
	set(CONFIG_CMD cmake -DCMAKE_INSTALL_PREFIX=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_MODULE_PATH:PATH=${CMAKE_CURRENT_SOURCE_DIR} -DETH_DEPENDENCY_INSTALL_DIR:PATH=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_BUILD_TYPE=None -DCMAKE_FIND_FRAMEWORK=LAST -Wno-dev .)
else()
    set(CONFIG_CMD cmake -DCMAKE_INSTALL_PREFIX=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_MODULE_PATH:PATH=${CMAKE_CURRENT_SOURCE_DIR} -DETH_DEPENDENCY_INSTALL_DIR:PATH=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_BUILD_TYPE=None -DCMAKE_FIND_FRAMEWORK=LAST .)
endif()


ExternalProject_Add(json-rpc-cpp
    # DEPENDS curl # re-enable later, when we build curl again
	GIT_REPOSITORY https://github.com/gogo40/libjson-rpc-cpp.git
	GIT_TAG 27f5da7a70c7a82b0614982cac829d6fd5fc8314 # this is roughly verson 0.3.2
	BINARY_DIR json-rpc-cpp-prefix/src/json-rpc-cpp
    CONFIGURE_COMMAND ${CONFIG_CMD}
	BUILD_COMMAND make -j 3
	INSTALL_COMMAND make install
)
