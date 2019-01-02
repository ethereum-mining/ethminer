file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.cpp")

find_package(Threads)

hunter_add_package(ZLIB)
find_package(ZLIB CONFIG REQUIRED)

add_library(devcore ${SOURCES} ${HEADERS})
target_link_libraries(devcore PUBLIC Boost::boost Boost::system)
target_link_libraries(devcore PRIVATE Threads::Threads)
target_link_libraries(devcore PUBLIC ZLIB::zlib)
