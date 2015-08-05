cmake_policy(SET CMP0015 NEW)
# let cmake autolink dependencies on windows
cmake_policy(SET CMP0020 NEW)
# this policy was introduced in cmake 3.0
# remove if, once 3.0 will be used on unix
if (${CMAKE_MAJOR_VERSION} GREATER 2)
	cmake_policy(SET CMP0043 OLD)
endif()

if (("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.6") AND NOT APPLE)
	# Supress warnings for qt headers for clang+ccache
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
endif ()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(BEFORE ${JSONCPP_INCLUDE_DIRS})
include_directories(BEFORE ..)
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

find_package (Qt5WebEngine)
find_package (Qt5WebEngineWidgets)
if (APPLE)
# TODO: remove indirect dependencies once macdeployqt is fixed
	find_package (Qt5WebEngineCore)
	find_package (Qt5DBus)
	find_package (Qt5PrintSupport)
endif()

qt5_wrap_ui(ui_Main.h Main.ui)
qt5_wrap_ui(ui_Connect.h Connect.ui)
qt5_wrap_ui(ui_Debugger.h Debugger.ui)
qt5_wrap_ui(ui_Transact.h Transact.ui)
qt5_wrap_ui(ui_ExportState.h ExportState.ui)
qt5_wrap_ui(ui_GetPassword.h GetPassword.ui)
qt5_wrap_ui(ui_GasPricing.h GasPricing.ui)

# Extensions
qt5_wrap_ui(ui_AllAccounts.h AllAccounts.ui)
qt5_wrap_ui(ui_LogPanel.h LogPanel.ui)

file(GLOB HEADERS "*.h")

if (APPLE)
	set(EXECUTABLE AlethZero)
else ()
	set(EXECUTABLE alethzero)
endif ()
	
# eth_add_executable is defined in cmake/EthExecutableHelper.cmake
eth_add_executable(${EXECUTABLE} 
	ICON alethzero
	UI_RESOURCES alethzero.icns Main.ui Connect.ui Debugger.ui Transact.ui ExportState.ui GetPassword.ui GasPricing.ui AllAccounts.ui LogPanel.ui
	WIN_RESOURCES alethzero.rc
)

add_dependencies(${EXECUTABLE} BuildInfo.h)

target_link_libraries(${EXECUTABLE} Qt5::Core)
target_link_libraries(${EXECUTABLE} Qt5::Widgets)
target_link_libraries(${EXECUTABLE} Qt5::WebEngine)
target_link_libraries(${EXECUTABLE} Qt5::WebEngineWidgets)
if (APPLE)
	target_link_libraries(${EXECUTABLE} Qt5::WebEngineCore)
	target_link_libraries(${EXECUTABLE} Qt5::DBus)
	target_link_libraries(${EXECUTABLE} Qt5::PrintSupport)
endif()
target_link_libraries(${EXECUTABLE} webthree)
target_link_libraries(${EXECUTABLE} ethereum)
target_link_libraries(${EXECUTABLE} evm)
target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} lll)
if (SOLIDITY)
	target_link_libraries(${EXECUTABLE} solidity)
endif ()
target_link_libraries(${EXECUTABLE} evmasm)
target_link_libraries(${EXECUTABLE} web3jsonrpc)
target_link_libraries(${EXECUTABLE} jsqrc)
target_link_libraries(${EXECUTABLE} natspec)
target_link_libraries(${EXECUTABLE} ${MHD_LIBRARIES})

if (SERPENT)
	target_link_libraries(${EXECUTABLE} serpent)
endif()

# eth_install_executable is defined in cmake/EthExecutableHelper.cmake
eth_install_executable(${EXECUTABLE} DLLS MHD_DLLS EVMJIT_DLLS OpenCL_DLLS)

