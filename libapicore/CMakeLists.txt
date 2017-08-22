set(SOURCES
    Api.h Api.cpp
    ApiServer.h ApiServer.cpp
	connectors/tcpsocketserver.h connectors/tcpsocketserver.cpp
)
set (LIBS 
    devcore
	libjson-rpc-cpp::server
)

if (WIN32)
	list(APPEND SOURCES "connectors/windowstcpsocketserver.h")
	list(APPEND SOURCES "connectors/windowstcpsocketserver.cpp")
	list(APPEND LIBS "ws2_32")
endif()

if(UNIX)
	list(APPEND SOURCES "connectors/linuxtcpsocketserver.h")
	list(APPEND SOURCES "connectors/linuxtcpsocketserver.cpp")
endif()

add_library(apicore ${SOURCES})
target_link_libraries(apicore ${LIBS})
target_include_directories(apicore PRIVATE ..)
