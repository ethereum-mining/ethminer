cmake_policy(SET CMP0015 NEW)

set(CMAKE_ASM_COMPILER "yasm")

set(EXECUTABLE secp256k1)

file(GLOB HEADERS "*.h") 

if ("${TARGET_PLATFORM}" STREQUAL "w64")
	add_custom_command(
        	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o
        	COMMAND yasm -f win64 -o ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o ${CMAKE_CURRENT_SOURCE_DIR}/field_5x52_asm.asm
	)
	add_custom_target(
        	asmOut ALL
        	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o
	)
	if(ETH_STATIC)
		add_library(${EXECUTABLE} STATIC ${EXECUTABLE}.c ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o)
	else()
		add_library(${EXECUTABLE} SHARED ${EXECUTABLE}.c ${CMAKE_CURRENT_BINARY_DIR}/field_5x52_asm.o)
	endif()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -DUSE_FIELD_5X52 -DUSE_FIELD_5X52_ASM -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM")
elseif(APPLE)
	if(ETH_STATIC)
		add_library(${EXECUTABLE} STATIC ${EXECUTABLE}.c field_5x52_asm.asm)
	else()
		find_library(GMP_LS gmp /usr/local/lib)
		add_library(${EXECUTABLE} SHARED ${EXECUTABLE}.c field_5x52_asm.asm)
	endif()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -DUSE_FIELD_GMP -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM")
else()
	if(ETH_STATIC)
		add_library(${EXECUTABLE} STATIC ${EXECUTABLE}.c field_5x52_asm.asm)
	else()
		add_library(${EXECUTABLE} SHARED ${EXECUTABLE}.c field_5x52_asm.asm)
	endif()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -DUSE_FIELD_GMP -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM")
endif()

if (NOT GMP_LS)
	set(GMP_LS gmp)
endif ()

target_link_libraries(${EXECUTABLE} ${GMP_LS})

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
