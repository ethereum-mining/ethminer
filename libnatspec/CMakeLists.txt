cmake_policy(SET CMP0015 NEW)
# let cmake autolink dependencies on windows
cmake_policy(SET CMP0020 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	cmake_policy(SET CMP0043 OLD)
endif()
set(CMAKE_AUTOMOC OFF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE natspec)

file(GLOB HEADERS "*.h")

qt5_add_resources(NATSPECQRC natspec.qrc)

add_library(${EXECUTABLE} ${RESOURCE_ADDED} ${SRC_LIST} ${HEADERS} ${NATSPECQRC})

target_link_libraries(${EXECUTABLE} Qt5::Core)
target_link_libraries(${EXECUTABLE} Qt5::Qml)
target_link_libraries(${EXECUTABLE} devcore)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
