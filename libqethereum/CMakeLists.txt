cmake_policy(SET CMP0015 OLD)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(..)
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})

set(EXECUTABLE qethereum)

file(GLOB HEADERS "*.h")

if(ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${RESOURCE_ADDED} ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${RESOURCE_ADDED} ${SRC_LIST} ${HEADERS})
endif()

qt5_use_modules(${EXECUTABLE} Core Gui WebKit WebKitWidgets Widgets Network Quick Qml)
target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} secp256k1)
target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_COMMON_LIBRARY})
target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_SERVER_LIBRARY})

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
