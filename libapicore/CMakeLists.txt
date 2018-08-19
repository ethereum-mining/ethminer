set(SOURCES
    ApiServer.h ApiServer.cpp
	httpServer.cpp httpServer.h
)

hunter_add_package(mongoose)
find_package(mongoose CONFIG REQUIRED)

add_library(apicore ${SOURCES})
target_link_libraries(apicore PRIVATE ethcore devcore ethminer-buildinfo mongoose::mongoose Boost::filesystem)
target_include_directories(apicore PRIVATE ..)
