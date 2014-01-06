cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)
include_directories(../../secp256k1/include)
include_directories(../libethereum)
link_directories(../libethereum)
link_directories(../../secp256k1)

add_executable(eth ${SRC_LIST})

target_link_libraries(eth secp256k1)
target_link_libraries(eth ethereum)
target_link_libraries(eth gmp)
