set(SOURCES
    Api.h Api.cpp
    ApiServer.h ApiServer.cpp
	httpServer.cpp httpServer.h
)

hunter_add_package(mongoose)
find_package(mongoose CONFIG REQUIRED)

add_library(apicore ${SOURCES})
target_link_libraries(apicore PRIVATE ethcore devcore ethminer-buildinfo libjson-rpc-cpp::server mongoose::mongoose)
target_include_directories(apicore PRIVATE ..)
