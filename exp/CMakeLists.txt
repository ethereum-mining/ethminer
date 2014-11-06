cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(..)

set(EXECUTABLE exp)

add_executable(${EXECUTABLE} ${SRC_LIST})

target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} p2p)
target_link_libraries(${EXECUTABLE} gmp)
if(MINIUPNPC_LS)
target_link_libraries(${EXECUTABLE} ${MINIUPNPC_LS})
endif()
target_link_libraries(${EXECUTABLE} ${LEVELDB_LS})

if ("${TARGET_PLATFORM}" STREQUAL "w64")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
	target_link_libraries(${EXECUTABLE} gcc)
	target_link_libraries(${EXECUTABLE} gdi32)
	target_link_libraries(${EXECUTABLE} ws2_32)
	target_link_libraries(${EXECUTABLE} mswsock)
	target_link_libraries(${EXECUTABLE} shlwapi)
	target_link_libraries(${EXECUTABLE} iphlpapi)
	target_link_libraries(${EXECUTABLE} boost_system-mt-s)
	target_link_libraries(${EXECUTABLE} boost_filesystem-mt-s)
	target_link_libraries(${EXECUTABLE} boost_thread_win32-mt-s)
	set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
elseif (UNIX)
else ()
	target_link_libraries(${EXECUTABLE} boost_system)
	target_link_libraries(${EXECUTABLE} boost_filesystem)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
endif ()

install( TARGETS ${EXECUTABLE} DESTINATION bin )

