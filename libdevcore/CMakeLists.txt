cmake_policy(SET CMP0015 NEW)

if (CMAKE_MAJOR_VERSION GREATER 1 AND CMAKE_MINOR_VERSION GREATER 7 AND CMAKE_PATCH_VERSION GREATER 11)
	cmake_policy(SET CMP0022 NEW)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)

set(EXECUTABLE devcore)

file(GLOB HEADERS "*.h")
if(ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})
endif()

include_directories(..)

if("${TARGET_PLATFORM}" STREQUAL "w64")
	include_directories(/usr/x86_64-w64-mingw32/include/cryptopp)
	target_link_libraries(${EXECUTABLE} boost_system-mt-s)
	target_link_libraries(${EXECUTABLE} boost_thread_win32-mt-s)
	target_link_libraries(${EXECUTABLE} iphlpapi)
	target_link_libraries(${EXECUTABLE} ws2_32)
	target_link_libraries(${EXECUTABLE} mswsock)
	target_link_libraries(${EXECUTABLE} shlwapi)
elseif (APPLE)
	# Latest mavericks boost libraries only come with -mt
	target_link_libraries(${EXECUTABLE} boost_system-mt)
	target_link_libraries(${EXECUTABLE} boost_filesystem-mt)
	target_link_libraries(${EXECUTABLE} boost_thread-mt)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
elseif (UNIX)
	find_package(Boost 1.53 REQUIRED COMPONENTS thread system)
	target_link_libraries(${EXECUTABLE} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY})
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
else ()
	target_link_libraries(${EXECUTABLE} boost_thread)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
endif ()

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

