file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.cpp")

find_package(Threads)
hunter_add_package(Boost COMPONENTS log thread)
find_package(Boost CONFIG REQUIRED log thread)
add_library(devcore ${SOURCES} ${HEADERS})
target_link_libraries(devcore PUBLIC Boost::boost Boost::system Boost::log Boost::thread)
target_link_libraries(devcore PRIVATE Threads::Threads)


