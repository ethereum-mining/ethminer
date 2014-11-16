# json-rpc-cpp is under heavy development, and multiplatform builds are not yet available. All the platforms currently need patches to make them work.

if(APPLE)
    set(PATCH_CMD patch -d src/example < ${CMAKE_CURRENT_SOURCE_DIR}/json-rpc-cpp_osx.patch)
	set(CONFIG_CMD cmake -DCMAKE_INSTALL_PREFIX=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_MODULE_PATH:PATH=${CMAKE_CURRENT_SOURCE_DIR} -DETH_DEPENDENCY_INSTALL_DIR:PATH=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_BUILD_TYPE=None -DCMAKE_FIND_FRAMEWORK=LAST -Wno-dev .)
else()
    set(PATCH_CMD patch --input=${CMAKE_CURRENT_SOURCE_DIR}/json-rpc-cpp_linux.patch --strip=1)
    set(CONFIG_CMD cmake -DCMAKE_INSTALL_PREFIX=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_MODULE_PATH:PATH=${CMAKE_CURRENT_SOURCE_DIR} -DETH_DEPENDENCY_INSTALL_DIR:PATH=${ETH_DEPENDENCY_INSTALL_DIR} -DCMAKE_BUILD_TYPE=None -DCMAKE_FIND_FRAMEWORK=LAST .)
endif()


ExternalProject_Add(json-rpc-cpp
	DEPENDS curl
	GIT_REPOSITORY https://github.com/cinemast/libjson-rpc-cpp.git 
	GIT_TAG 0.2.1
	BINARY_DIR json-rpc-cpp-prefix/src/json-rpc-cpp
    PATCH_COMMAND ${PATCH_CMD}
    CONFIGURE_COMMAND ${CONFIG_CMD}
	BUILD_COMMAND make jsonrpc -j 3
	INSTALL_COMMAND make install
)
