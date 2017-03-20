# all dependencies that are not directly included in the cpp-ethereum distribution are defined here
# for this to work, download the dependency via the cmake script in extdep or install them manually!

# by defining this variable, cmake will look for dependencies first in our own repository before looking in system paths like /usr/local/ ...
# this must be set to point to the same directory as $ETH_DEPENDENCY_INSTALL_DIR in /extdep directory
string(TOLOWER ${CMAKE_SYSTEM_NAME} _system_name)
if (CMAKE_CL_64)
	set (ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extdep/install/${_system_name}/x64")
else ()
	set (ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extdep/install/${_system_name}/Win32")
endif()
set (CMAKE_PREFIX_PATH ${ETH_DEPENDENCY_INSTALL_DIR})

# setup directory for cmake generated files and include it globally
# it's not used yet, but if we have more generated files, consider moving them to ETH_GENERATED_DIR
set(ETH_GENERATED_DIR "${PROJECT_BINARY_DIR}/gen")
include_directories(${ETH_GENERATED_DIR})

# custom cmake scripts
set(ETH_SCRIPTS_DIR ${CMAKE_CURRENT_LIST_DIR}/scripts)

# Qt5 requires opengl
# TODO use proper version of windows SDK (32 vs 64)
# TODO make it possible to use older versions of windows SDK (7.0+ should also work)
# TODO it windows SDK is NOT FOUND, throw ERROR
# from https://github.com/rpavlik/cmake-modules/blob/master/FindWindowsSDK.cmake
if (WIN32)
	find_package(WINDOWSSDK REQUIRED)
	message(" - WindowsSDK dirs: ${WINDOWSSDK_DIRS}")
	set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${WINDOWSSDK_DIRS})
endif()

# homebrew installs qts in opt
if (APPLE)
	set (CMAKE_PREFIX_PATH "/usr/local/opt/qt5" ${CMAKE_PREFIX_PATH})
	set (CMAKE_PREFIX_PATH "/usr/local/opt/v8-315" ${CMAKE_PREFIX_PATH})
endif()

find_program(CTEST_COMMAND ctest)
message(STATUS "ctest path: ${CTEST_COMMAND}")

# Dependencies must have a version number, to ensure reproducible build. The version provided here is the one that is in the extdep repository. If you use system libraries, version numbers may be different.

find_package (CryptoPP 5.6.2 REQUIRED)
message(" - CryptoPP header: ${CRYPTOPP_INCLUDE_DIRS}")
message(" - CryptoPP lib   : ${CRYPTOPP_LIBRARIES}")

# TODO the Jsoncpp package does not yet check for correct version number
find_package (Jsoncpp 0.60 REQUIRED)
message(" - Jsoncpp header: ${JSONCPP_INCLUDE_DIRS}")
message(" - Jsoncpp lib   : ${JSONCPP_LIBRARIES}")

find_package (json_rpc_cpp 0.4 REQUIRED)
message (" - json-rpc-cpp header: ${JSON_RPC_CPP_INCLUDE_DIRS}")
message (" - json-rpc-cpp lib   : ${JSON_RPC_CPP_LIBRARIES}")

# curl is only requried for tests
# TODO specify min curl version, on windows we are currently using 7.29
find_package (CURL)
message(" - curl header: ${CURL_INCLUDE_DIRS}")
message(" - curl lib   : ${CURL_LIBRARIES}")

find_package (OpenCL)
if (OpenCL_FOUND)
	message(" - opencl header: ${OpenCL_INCLUDE_DIRS}")
	message(" - opencl lib   : ${OpenCL_LIBRARIES}")
endif()

find_package (CUDA)
if (CUDA_FOUND)
	message(" - CUDA header: ${CUDA_INCLUDE_DIRS}")
	message(" - CUDA lib   : ${CUDA_LIBRARIES}")
endif()

# find location of jsonrpcstub
find_program(ETH_JSON_RPC_STUB jsonrpcstub)
message(" - jsonrpcstub location    : ${ETH_JSON_RPC_STUB}")

# use multithreaded boost libraries, with -mt suffix
set(Boost_USE_MULTITHREADED ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

# TODO hanlde other msvc versions or it will fail find them
	set(Boost_COMPILER -vc120)
# use static boost libraries *.lib
	set(Boost_USE_STATIC_LIBS ON)

elseif (APPLE)

# use static boost libraries *.a
	set(Boost_USE_STATIC_LIBS ON)

elseif (UNIX)
# use dynamic boost libraries .dll
	set(Boost_USE_STATIC_LIBS OFF)

endif()

find_package(Boost 1.54.0 REQUIRED COMPONENTS thread date_time system regex chrono filesystem unit_test_framework program_options random)

message(" - boost header: ${Boost_INCLUDE_DIRS}")
message(" - boost lib   : ${Boost_LIBRARIES}")

if (APPLE)
	link_directories(/usr/local/lib)
	include_directories(/usr/local/include)
endif()
