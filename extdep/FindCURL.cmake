# hacky way to resolve nested dependencies
find_library(CURL_LIBRARY NAMES curl
	PATHS
	${ETH_DEPENDENCY_INSTALL_DIR}/lib
	)

set(CURL_LIBRARIES ${CURL_LIBRARY})
set(CURL_INCLUDE_DIRS ${ETH_DEPENDENCY_INSTALL_DIR}/include)

