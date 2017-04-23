include_directories(..)
include_directories(${JSONCPP_INCLUDE_DIRS})

set(SOURCES
    EthStratumClient.h EthStratumClient.cpp
    EthStratumClientV2.h EthStratumClientV2.cpp
)

add_library(ethstratum ${SOURCES})
target_link_libraries(ethstratum PUBLIC Boost::system)
target_include_directories(ethstratum PUBLIC SYSTEM ${JSONCPP_INCLUDE_DIRS})
