cmake_policy(SET CMP0015 NEW)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)

if(NOT APPLE)
	add_library(ethereum ${SRC_LIST})
else()
	add_library(ethereum SHARED ${SRC_LIST})
endif()

file(GLOB HEADERS "*.h") 

include_directories(../secp256k1)

target_link_libraries(ethereum secp256k1)
target_link_libraries(ethereum miniupnpc)
target_link_libraries(ethereum leveldb)
target_link_libraries(ethereum gmp)

if(${TARGET_PLATFORM} STREQUAL "w64")
	include_directories(/usr/x86_64-w64-mingw32/include/cryptopp)
	target_link_libraries(ethereum cryptopp)
	target_link_libraries(ethereum boost_system-mt-s)
	target_link_libraries(ethereum boost_filesystem-mt-s)
	target_link_libraries(ethereum boost_thread_win32-mt-s)
	target_link_libraries(ethereum iphlpapi)
	target_link_libraries(ethereum ws2_32)
	target_link_libraries(ethereum mswsock)
	target_link_libraries(ethereum shlwapi)
else ()
	target_link_libraries(ethereum ${CRYPTOPP_LIBRARIES})
	target_link_libraries(ethereum boost_system)
	target_link_libraries(ethereum boost_filesystem)
	find_package(Threads REQUIRED)
	target_link_libraries(ethereum ${CMAKE_THREAD_LIBS_INIT})
endif ()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	include_directories(/usr/local/include)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

message("Installation path: ${CMAKE_INSTALL_PREFIX}")

install( TARGETS ethereum ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/ethereum )

