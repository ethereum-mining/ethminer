cmake_policy(SET CMP0015 NEW)

set(CMAKE_ASM_COMPILER "yasm")

#aux_source_directory(. SRC_LIST)
if(NOT APPLE)
	add_library(secp256k1 secp256k1.c field_5x52_asm.asm)
else()
	add_library(secp256k1 SHARED secp256k1.c field_5x52_asm.asm)
endif()

#set(CMAKE_C_FLAGS "-DUSE_FIELD_5X52 -DUSE_FIELD_5X52_ASM -DUSE_NUM_OPENSSL -DUSE_FIELD_INV_BUILTIN")
#target_link_libraries(secp256k1 crypto)
set(CMAKE_C_FLAGS "-std=c99 -DUSE_FIELD_GMP -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM")
target_link_libraries(secp256k1 gmp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	include_directories(/usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")