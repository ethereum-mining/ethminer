set(SOURCES
	EthashAux.h EthashAux.cpp
	Farm.cpp Farm.h
	Miner.h Miner.cpp
)

include_directories(BEFORE ..)

add_library(ethcore ${SOURCES})
target_link_libraries(ethcore PUBLIC devcore ethash::ethash Boost::filesystem Boost::thread PRIVATE hwmon)

if(OPENCL)
	target_link_libraries(ethcore PRIVATE ethash-cl)
endif()
if(CUDA)
	target_link_libraries(ethcore PUBLIC ethash-cuda)
endif()
if(CPU)
	target_link_libraries(ethcore PUBLIC ethash-cpu)
endif()
