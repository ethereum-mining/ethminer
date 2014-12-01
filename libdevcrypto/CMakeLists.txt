cmake_policy(SET CMP0015 OLD)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

set(EXECUTABLE devcrypto)

file(GLOB HEADERS "*.h")
if(ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})
endif()

include_directories(${CRYPTOPP_INCLUDE_DIR})
include_directories(${LEVELDB_INCLUDE_DIR})
include_directories(..) 

target_link_libraries(${EXECUTABLE} ${LEVELDB_LIBRARY})
target_link_libraries(${EXECUTABLE} ${CRYPTOPP_LIBRARIES})
target_link_libraries(${EXECUTABLE} devcore)
target_link_libraries(${EXECUTABLE} secp256k1)
target_link_libraries(${EXECUTABLE} gmp)

if (APPLE)
	# Latest mavericks boost libraries only come with -mt
	target_link_libraries(${EXECUTABLE} boost_filesystem-mt)
elseif (UNIX)
	find_package(Boost 1.53 REQUIRED COMPONENTS filesystem)
	target_link_libraries(${EXECUTABLE} ${Boost_FILESYSTEM_LIBRARY})
endif ()

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

