cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

set(EXECUTABLE pyserpent)

# set(CMAKE_INSTALL_PREFIX ../lib)
add_library(${EXECUTABLE} SHARED ${SRC_LIST})

if (UNIX)
	FIND_PACKAGE(Boost 1.53 REQUIRED COMPONENTS python)
endif()
file(GLOB HEADERS "*.h") 

include_directories(..)

add_definitions(-DETH_PYTHON)
include_directories(${PYTHON_ID})
target_link_libraries(${EXECUTABLE} ${PYTHON_LS})

target_link_libraries(${EXECUTABLE} serpent)
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} evmface)
target_link_libraries(${EXECUTABLE} ethential)
target_link_libraries(${EXECUTABLE} gmp)
target_link_libraries(${EXECUTABLE} gmp)

#g++ $(CXXFLAGS) -shared $(PLATFORM_OPTS) $(TARGET).o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) $(COMMON_OBJS) -o $(TARGET).so

if(${TARGET_PLATFORM} STREQUAL "w64")
	target_link_libraries(${EXECUTABLE} boost_python_win32-mt-s)
	target_link_libraries(${EXECUTABLE} boost_thread_win32-mt-s)
	target_link_libraries(${EXECUTABLE} iphlpapi)
	target_link_libraries(${EXECUTABLE} ws2_32)
	target_link_libraries(${EXECUTABLE} mswsock)
	target_link_libraries(${EXECUTABLE} shlwapi)
elseif (APPLE)
	# Latest mavericks boost libraries only come with -mt
	target_link_libraries(${EXECUTABLE} boost_python-mt)
	target_link_libraries(${EXECUTABLE} boost_thread-mt)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
elseif (UNIX)
	target_link_libraries(${EXECUTABLE} ${Boost_PYTHON_LIBRARY})
	target_link_libraries(${EXECUTABLE} ${Boost_THREAD_LIBRARY})
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
else ()
	target_link_libraries(${EXECUTABLE} boost_python)
	target_link_libraries(${EXECUTABLE} boost_thread)
	find_package(Threads REQUIRED)
	target_link_libraries(${EXECUTABLE} ${CMAKE_THREAD_LIBS_INIT})
endif ()

message("Installation path: ${CMAKE_INSTALL_PREFIX}")

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

