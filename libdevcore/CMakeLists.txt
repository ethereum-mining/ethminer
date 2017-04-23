cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SOURCES)

include_directories(BEFORE ${JSONCPP_INCLUDE_DIRS})
include_directories(BEFORE ..)

set(EXECUTABLE devcore)

file(GLOB HEADERS "*.h")

add_library(devcore ${SOURCES} ${HEADERS})

target_link_libraries(devcore PUBLIC ${JSONCPP_LIBRARIES} Boost::boost)

if (APPLE)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
elseif (UNIX)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endif()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

