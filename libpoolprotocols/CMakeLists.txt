set(SOURCES
	PoolClient.h PoolClient.cpp
	PoolManager.h PoolManager.cpp
	testing/SimulateClient.h testing/SimulateClient.cpp
    stratum/EthStratumClient.h stratum/EthStratumClient.cpp
    getwork/EthGetworkClient.h getwork/EthGetworkClient.cpp getwork/jsonrpc_getwork.h
)

hunter_add_package(OpenSSL)
find_package(OpenSSL REQUIRED)

add_library(poolprotocols ${SOURCES})
target_link_libraries(poolprotocols libjson-rpc-cpp::client)
target_link_libraries(poolprotocols devcore Boost::system jsoncpp_lib_static)
target_link_libraries(poolprotocols OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(poolprotocols PRIVATE ..)
