cmake_policy(SET CMP0015 NEW)
# let cmake autolink dependencies on windows
cmake_policy(SET CMP0020 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	# old policy do not use MACOSX_RPATH
	cmake_policy(SET CMP0042 OLD)
	cmake_policy(SET CMP0043 OLD)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(..)

set(EXECUTABLE qwebthree)

file(GLOB HEADERS "*.h")

if(ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${RESOURCE_ADDED} ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${RESOURCE_ADDED} ${SRC_LIST} ${HEADERS})
endif()

target_link_libraries(${EXECUTABLE} Qt5::Core)
target_link_libraries(${EXECUTABLE} Qt5::Gui)
target_link_libraries(${EXECUTABLE} Qt5::WebKit)
target_link_libraries(${EXECUTABLE} Qt5::WebKitWidgets)
target_link_libraries(${EXECUTABLE} Qt5::Widgets)
target_link_libraries(${EXECUTABLE} Qt5::Network)
target_link_libraries(${EXECUTABLE} Qt5::Quick)
target_link_libraries(${EXECUTABLE} Qt5::Qml)
target_link_libraries(${EXECUTABLE} ${JSON_RPC_CPP_SERVER_LIBRARIES})
target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} secp256k1)

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
