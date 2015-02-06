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
set(CMAKE_AUTOMOC OFF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(..)

set(EXECUTABLE natspec)

file(GLOB HEADERS "*.h")

qt5_add_resources(NATSPECQRC natspec.qrc)

if (ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${RESOURCE_ADDED} ${SRC_LIST} ${HEADERS} ${NATSPECQRC})
else()
	add_library(${EXECUTABLE} SHARED ${RESOURCE_ADDED} ${SRC_LIST} ${HEADERS} ${NATSPECQRC})
endif()

target_link_libraries(${EXECUTABLE} Qt5::Core)
target_link_libraries(${EXECUTABLE} Qt5::Qml)
target_link_libraries(${EXECUTABLE} devcore)

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
