cmake_policy(SET CMP0015 OLD)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(${CRYPTOPP_INCLUDE_DIR})
include_directories(${LEVELDB_INCLUDE_DIR})
include_directories(..) 

set(EXECUTABLE devcrypto)

file(GLOB HEADERS "*.h")
if(ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})
endif()

target_link_libraries(${EXECUTABLE} ${Boost_FILESYSTEM_LIBRARY_RELEASE})
target_link_libraries(${EXECUTABLE} ${LEVELDB_LIBRARY})
target_link_libraries(${EXECUTABLE} ${CRYPTOPP_LIBRARIES})
target_link_libraries(${EXECUTABLE} devcore)
target_link_libraries(${EXECUTABLE} secp256k1)

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

