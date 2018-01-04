file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.cpp")

find_package(Threads)

add_library(devcore ${SOURCES} ${HEADERS})
target_link_libraries(devcore PUBLIC Boost::boost Boost::system)
target_link_libraries(devcore PRIVATE Threads::Threads)
