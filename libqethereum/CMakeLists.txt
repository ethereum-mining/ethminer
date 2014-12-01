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
target_link_libraries(${EXECUTABLE} ethereum secp256k1 ${JSON_RPC_CPP_LIBRARIES})

