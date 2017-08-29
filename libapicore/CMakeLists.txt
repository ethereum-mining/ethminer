set(SOURCES
    Api.h Api.cpp
    ApiServer.h ApiServer.cpp
)

add_library(apicore ${SOURCES})
target_link_libraries(apicore devcore libjson-rpc-cpp::server)
target_include_directories(apicore PRIVATE ..)
