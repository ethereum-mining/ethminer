cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(../libethereum)
link_directories(../libethereum)

add_executable(eth ${SRC_LIST})

find_package(Threads REQUIRED)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	include_directories(/usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

target_link_libraries(eth ethereum)
target_link_libraries(eth miniupnpc)
target_link_libraries(eth leveldb)
target_link_libraries(eth ${CRYPTOPP_LIBRARIES})
target_link_libraries(eth gmp)
target_link_libraries(eth boost_system)
target_link_libraries(eth boost_filesystem)
target_link_libraries(eth ${CMAKE_THREAD_LIBS_INIT})

install( TARGETS eth DESTINATION bin )

