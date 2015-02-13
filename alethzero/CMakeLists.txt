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
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})

qt5_wrap_ui(ui_Main.h Main.ui)

file(GLOB HEADERS "*.h")

if (APPLE)
	set(EXECUTABLE AlethZero)
else ()
	set(EXECUTABLE alethzero)
endif ()
	
# eth_add_executable is defined in cmake/EthExecutableHelper.cmake
eth_add_executable(${EXECUTABLE} 
	ICON alethzero 
	UI_RESOURCES alethzero.icns Main.ui
	WIN_RESOURCES alethzero.rc
)

add_dependencies(${EXECUTABLE} BuildInfo.h)

target_link_libraries(${EXECUTABLE} Qt5::Core)
target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} qwebthree)
target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} evm)
target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} secp256k1)
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} solidity)
target_link_libraries(${EXECUTABLE} evmcore)
target_link_libraries(${EXECUTABLE} devcore)
target_link_libraries(${EXECUTABLE} web3jsonrpc)
target_link_libraries(${EXECUTABLE} jsqrc)
target_link_libraries(${EXECUTABLE} natspec)

if (NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
	target_link_libraries(${EXECUTABLE} serpent)
endif()

# eth_install_executable is defined in cmake/EthExecutableHelper.cmake
eth_install_executable(${EXECUTABLE})

