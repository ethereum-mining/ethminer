cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

if(NOT APPLE)
	add_library(ethereum ${SRC_LIST})
else()
	add_library(ethereum SHARED ${SRC_LIST})
endif()

file(GLOB HEADERS "*.h") 

find_package(Threads REQUIRED)

include_directories(../secp256k1)

target_link_libraries(ethereum secp256k1)
target_link_libraries(ethereum miniupnpc)
target_link_libraries(ethereum leveldb)
target_link_libraries(ethereum ${CRYPTOPP_LIBRARIES})
target_link_libraries(ethereum gmp)
target_link_libraries(ethereum boost_system)
target_link_libraries(ethereum boost_filesystem)
target_link_libraries(ethereum ${CMAKE_THREAD_LIBS_INIT})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	include_directories(/usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

message("Installation path: ${CMAKE_INSTALL_PREFIX}")

install( TARGETS ethereum ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/ethereum )

