set(SOURCES
	BlockHeader.h BlockHeader.cpp
	EthashAux.h EthashAux.cpp
	Exceptions.h
	Farm.h
	Miner.h Miner.cpp
)


include_directories(BEFORE ..)

add_library(ethcore ${SOURCES})
target_link_libraries(ethcore ethash devcore)

if(ETHASHCL)
	target_link_libraries(ethcore ethash-cl)
endif()
if(ETHASHCUDA)
	target_link_libraries(ethcore ethash-cuda)
endif()
