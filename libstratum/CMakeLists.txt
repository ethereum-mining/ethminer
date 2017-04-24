set(SOURCES
    EthStratumClient.h EthStratumClient.cpp
    EthStratumClientV2.h EthStratumClientV2.cpp
)

add_library(ethstratum ${SOURCES})
target_link_libraries(ethstratum PUBLIC Boost::system jsoncpp_lib_static)
target_include_directories(ethstratum PRIVATE ..)
