file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.cpp")

add_library(ethcore ${SOURCES} ${HEADERS})

find_package(Boost COMPONENTS system thread REQUIRED)

target_link_libraries(ethcore
  PUBLIC
  ${Boost_SYSTEM_LIBRARY}
	${Boost_THREAD_LIBRARY}
)

target_link_libraries(ethcore PUBLIC devcore ethash::ethash PRIVATE hwmon)

if(ETHASHCL)
	target_link_libraries(ethcore PRIVATE ethash-cl)
endif()
if(ETHASHCUDA)
	target_link_libraries(ethcore PUBLIC ethash-cuda)
endif()
if(ETHASHCPU)
	target_link_libraries(ethcore PUBLIC ethash-cpu)
endif()
