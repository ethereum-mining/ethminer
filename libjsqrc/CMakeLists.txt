cmake_policy(SET CMP0015 NEW)
# let cmake autolink dependencies on windows
cmake_policy(SET CMP0020 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	cmake_policy(SET CMP0043 OLD)
endif()
set(CMAKE_AUTOMOC OFF)

qt5_add_resources(JSQRC js.qrc)
add_library(jsqrc STATIC ${JSQRC})
target_link_libraries(jsqrc Qt5::Core)

if (ETH_NODE AND ETH_NPM)
	add_custom_target(ethereumjs)
	add_custom_command(TARGET ethereumjs
		POST_BUILD
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMAND bash compilejs.sh ${ETH_NPM_DIRECTORY} ${ETH_NODE_DIRECTORY}
	)
	add_dependencies(jsqrc ethereumjs)
endif()

install( TARGETS jsqrc ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
