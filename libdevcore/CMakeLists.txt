file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.cpp")

add_library(devcore ${SOURCES} ${HEADERS})

find_package(Boost COMPONENTS system REQUIRED)
find_package(Threads)

target_link_libraries(devcore
  PUBLIC
  ${Boost_SYSTEM_LIBRARY}
)

target_link_libraries(devcore
  PRIVATE
  Threads::Threads
)
