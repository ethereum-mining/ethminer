set(SOURCES
	EthashAux.h EthashAux.cpp
	Farm.cpp Farm.h
	Miner.h Miner.cpp
)

include_directories(BEFORE ..)
include_directories(${jsoncpp_INCLUDE_DIR})
include_directories(${OpenCL_INCLUDE_DIR})

add_library(ethcore ${SOURCES})
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
