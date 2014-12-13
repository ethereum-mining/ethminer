cmake_policy(SET CMP0015 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	# old policy do not use MACOSX_RPATH
	cmake_policy(SET CMP0042 OLD)
endif()
set(CMAKE_AUTOMOC OFF)

set(CMAKE_ASM_COMPILER "yasm")

set(EXECUTABLE secp256k1)
file(GLOB HEADERS "*.h") 

if (APPLE OR UNIX)

	if (ETH_STATIC)
		add_library(${EXECUTABLE} STATIC ${EXECUTABLE}.c field_5x52_asm.asm)
	else()
		add_library(${EXECUTABLE} SHARED ${EXECUTABLE}.c field_5x52_asm.asm)
	endif()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -DUSE_FIELD_GMP -DUSE_NUM_GMP -DUSE_FIELD_INV_NUM")
	target_link_libraries(${EXECUTABLE} ${GMP_LIBRARIES})
else()

	include_directories(${Boost_INCLUDE_DIRS})

	if (ETH_STATIC)
		add_library(${EXECUTABLE} STATIC ${EXECUTABLE}.c)
	else()
		add_library(${EXECUTABLE} SHARED ${EXECUTABLE}.c)
	endif()
	# /TP - compile project as cpp project
	set_target_properties(${EXECUTABLE} PROPERTIES COMPILE_FLAGS "/TP")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_NUM_BOOST -DUSE_FIELD_10X26 -DUSE_FIELD_INV_BUILTIN")
	
endif()

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
