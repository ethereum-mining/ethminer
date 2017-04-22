cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SOURCES)

include_directories(BEFORE ${JSONCPP_INCLUDE_DIRS})
include_directories(BEFORE ..)
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE devcore)

file(GLOB HEADERS "*.h")

add_library(devcore ${SOURCES} ${HEADERS})

target_link_libraries(devcore ${Boost_THREAD_LIBRARIES})
target_link_libraries(devcore ${Boost_RANDOM_LIBRARIES})
target_link_libraries(devcore ${Boost_FILESYSTEM_LIBRARIES})
target_link_libraries(devcore ${Boost_SYSTEM_LIBRARIES})
target_link_libraries(devcore ${JSONCPP_LIBRARIES})

# transitive dependencies for windows executables
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	target_link_libraries(${EXECUTABLE} ${Boost_CHRONO_LIBRARIES})
	target_link_libraries(${EXECUTABLE} ${Boost_DATE_TIME_LIBRARIES})
endif()

if (APPLE)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
elseif (UNIX)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
endif()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

