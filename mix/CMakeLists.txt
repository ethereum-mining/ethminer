cmake_policy(SET CMP0015 NEW)
# let cmake autolink dependencies on windows
cmake_policy(SET CMP0020 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	cmake_policy(SET CMP0043 OLD)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

aux_source_directory(. SRC_LIST)

include_directories(..)
include_directories(BEFORE ${JSONCPP_INCLUDE_DIRS})

find_package (Qt5WebEngine QUIET)
qt5_add_resources(UI_RESOURCES res.qrc)

file(GLOB HEADERS "*.h")

set(EXECUTABLE mix)

if ("${Qt5WebEngine_VERSION_STRING}" VERSION_GREATER "5.3.0")
	set (ETH_HAVE_WEBENGINE TRUE)
	qt5_add_resources(UI_RESOURCES web.qrc)
else()
	qt5_add_resources(UI_RESOURCES noweb.qrc)
endif()

add_definitions(-DQT_QML_DEBUG)

# eth_add_executable is defined in cmake/EthExecutableHelper.cmake
eth_add_executable(${EXECUTABLE} 
	ICON mix
	UI_RESOURCES ${UI_RESOURCES}
)

target_link_libraries(${EXECUTABLE} Qt5::Core)
target_link_libraries(${EXECUTABLE} Qt5::Gui)
target_link_libraries(${EXECUTABLE} Qt5::Widgets)
target_link_libraries(${EXECUTABLE} Qt5::Network)
target_link_libraries(${EXECUTABLE} Qt5::Quick)
target_link_libraries(${EXECUTABLE} Qt5::Qml)
target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} evm)
target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} secp256k1)
if (NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
target_link_libraries(${EXECUTABLE} serpent)
endif()
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} solidity)
target_link_libraries(${EXECUTABLE} evmcore)
target_link_libraries(${EXECUTABLE} devcore)
target_link_libraries(${EXECUTABLE} jsqrc)
target_link_libraries(${EXECUTABLE} web3jsonrpc)

if (${ETH_HAVE_WEBENGINE})
	add_definitions(-DETH_HAVE_WEBENGINE)
	target_link_libraries(${EXECUTABLE} Qt5::WebEngine)
endif()

# eth_install_executable is defined in cmake/EthExecutableHelper.cmake
eth_install_executable(${EXECUTABLE}
	QMLDIR ${CMAKE_CURRENT_SOURCE_DIR}/qml
)

#add qml asnd stdc files to project tree in Qt creator
file(GLOB_RECURSE QMLFILES "qml/*.*")
file(GLOB_RECURSE SOLFILES "stdc/*.*")
add_custom_target(dummy SOURCES ${QMLFILES} ${SOLFILES})
