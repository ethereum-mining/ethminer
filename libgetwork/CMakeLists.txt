set(SOURCES
    EthGetworkClient.h EthGetworkClient.cpp
    jsonrpc_getwork.h
)

add_library(ethgetwork ${SOURCES})
target_link_libraries(ethgetwork libjson-rpc-cpp::client)
target_link_libraries(ethgetwork PUBLIC devcore Boost::system jsoncpp_lib_static)
target_include_directories(ethgetwork PRIVATE ..)
