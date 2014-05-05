cmake_policy(SET CMP0015 NEW)

set(CMAKE_ASM_COMPILER "yasm")

#aux_source_directory(. SRC_LIST)

if ("${TARGET_PLATFORM}" STREQUAL "w64")
	add_custom_command(
        	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o
        	COMMAND yasm -f win64 -o ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o ${CMAKE_CURRENT_SOURCE_DIR}/field_5x52_asm.asm
	)
	add_custom_target(
        	asmOut ALL
        	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o
	)
	add_library(secp256k1 secp256k1.c ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -DUSE_FIELD_5X52 -DUSE_FIELD_5X52_ASM -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM")
elseif(APPLE)
	# set(CMAKE_INSTALL_PREFIX ../lib)
	add_library(secp256k1 SHARED secp256k1.c field_5x52_asm.asm)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -DUSE_FIELD_GMP -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM")
else()
	add_library(secp256k1 secp256k1.c field_5x52_asm.asm)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -DUSE_FIELD_GMP -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM")
endif()

target_link_libraries(secp256k1 gmp)

