set(SOURCES
	PoolURI.cpp PoolURI.h
	PoolClient.h
	PoolManager.h PoolManager.cpp
	testing/SimulateClient.h testing/SimulateClient.cpp
	stratum/EthStratumClient.h stratum/EthStratumClient.cpp
	getwork/EthGetworkClient.h getwork/EthGetworkClient.cpp getwork/jsonrpc_getwork.h
)

hunter_add_package(OpenSSL)
find_package(OpenSSL REQUIRED)

add_library(poolprotocols ${SOURCES})
target_link_libraries(poolprotocols PRIVATE devcore ethminer-buildinfo ethash::ethash libjson-rpc-cpp::client Boost::system jsoncpp_lib_static OpenSSL::SSL OpenSSL::Crypto network-uri)
target_include_directories(poolprotocols PRIVATE ..)
