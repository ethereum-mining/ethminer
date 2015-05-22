cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})

if (JSCONSOLE)
	include_directories(${V8_INCLUDE_DIRS})
endif()

set(EXECUTABLE ethkey)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

add_dependencies(${EXECUTABLE} BuildInfo.h)

target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} ethcore)

if (DEFINED WIN32 AND NOT DEFINED CMAKE_COMPILER_IS_MINGW)
	eth_copy_dlls("${EXECUTABLE}" MHD_DLLS)
endif()

if (APPLE)
	install(TARGETS ${EXECUTABLE} DESTINATION bin)
else()
	eth_install_executable(${EXECUTABLE})
endif()
