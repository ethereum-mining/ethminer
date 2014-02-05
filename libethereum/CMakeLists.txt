cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)
add_library(ethereum ${SRC_LIST})
file(GLOB HEADERS "*.h") 

find_package(Threads REQUIRED)

target_link_libraries(ethereum secp256k1)
target_link_libraries(ethereum miniupnpc)
target_link_libraries(ethereum leveldb)
target_link_libraries(ethereum ${CRYPTOPP_LIBRARIES})
target_link_libraries(ethereum gmp)
target_link_libraries(ethereum boost_system)
target_link_libraries(ethereum boost_filesystem)
target_link_libraries(ethereum ${CMAKE_THREAD_LIBS_INIT})

message("Installation path: ${CMAKE_INSTALL_PREFIX}")

install( TARGETS ethereum ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/ethereum )

