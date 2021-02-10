set(SOURCES
<<<<<<< HEAD
	EthashAux.h EthashAux.cpp
	Farm.cpp Farm.h
	Miner.h Miner.cpp
=======
    BlockHeader.h BlockHeader.cpp
    EthashAux.h EthashAux.cpp
    Exceptions.h
    Farm.h
    Miner.h Miner.cpp
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
)

include_directories(BEFORE ..)

add_library(ethcore ${SOURCES})
target_link_libraries(ethcore PUBLIC devcore ethash::ethash PRIVATE hwmon)

if(ETHASHCL)
    target_link_libraries(ethcore PRIVATE ethash-cl)
endif()
if(ETHASOHCL)
    target_link_libraries(ethcore PRIVATE ethash-ocl)
endif()
if(ETHASHCUDA)
    target_link_libraries(ethcore PUBLIC ethash-cuda)
endif()
if(ETHASHCPU)
	target_link_libraries(ethcore PUBLIC ethash-cpu)
endif()
