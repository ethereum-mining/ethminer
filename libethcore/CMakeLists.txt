cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${Boost_INCLUDE_DIRS})

if (CPUID_FOUND)
	include_directories(${Cpuid_INCLUDE_DIRS})
endif ()

set(EXECUTABLE ethcore)

file(GLOB HEADERS "*.h")

add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

target_link_libraries(${EXECUTABLE} ethash)
target_link_libraries(${EXECUTABLE} evmcore)
if (ETHASHCL)
	target_link_libraries(${EXECUTABLE} ethash-cl)
endif ()
target_link_libraries(${EXECUTABLE} devcrypto)
if (CPUID_FOUND)
	target_link_libraries(${EXECUTABLE} ${CPUID_LIBRARIES})
endif ()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

