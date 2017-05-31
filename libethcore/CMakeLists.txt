file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")


include_directories(BEFORE ..)

add_library(ethcore ${SOURCES} ${HEADERS})
target_link_libraries(ethcore ethash devcore)

if(ETHASHCL)
	target_link_libraries(ethcore ethash-cl)
endif()
if(ETHASHCUDA)
	target_include_directories(ethcore PRIVATE ${CUDA_INCLUDE_DIRS})
	target_link_libraries(ethcore ethash-cuda)
endif()
