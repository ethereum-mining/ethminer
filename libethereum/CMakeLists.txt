cmake_policy(SET CMP0015 NEW)

include_directories(../../secp256k1/include)
link_directories(../../secp256k1)
aux_source_directory(. SRC_LIST)
add_library(ethereum ${SRC_LIST})

target_link_libraries(ethereum secp256k1)
target_link_libraries(ethereum gmp)
