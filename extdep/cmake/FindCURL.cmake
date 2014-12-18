# hacky way to resolve nested dependencies - needed for json-rpc-cpp
find_library(CURL_LIBRARY NAMES curl
	PATHS
	${ETH_DEPENDENCY_INSTALL_DIR}/lib
	)

set(CURL_LIBRARIES ${CURL_LIBRARY})
set(CURL_INCLUDE_DIRS ${ETH_DEPENDENCY_INSTALL_DIR}/include)

