cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..) 
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CRYPTOPP_INCLUDE_DIRS})
include_directories(${DB_INCLUDE_DIRS})

set(EXECUTABLE devcrypto)

file(GLOB HEADERS "*.h")
add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

target_link_libraries(${EXECUTABLE} ${Boost_FILESYSTEM_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${DB_LIBRARIES})
target_link_libraries(${EXECUTABLE} ${CRYPTOPP_LIBRARIES})
target_link_libraries(${EXECUTABLE} scrypt)
target_link_libraries(${EXECUTABLE} devcore)
if (NOT WIN32)
	add_definitions(-DETH_HAVE_SECP256K1)
	target_link_libraries(${EXECUTABLE} secp256k1)
endif ()

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

