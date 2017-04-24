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

find_program(CTEST_COMMAND ctest)
message(STATUS "ctest path: ${CTEST_COMMAND}")

find_package (json_rpc_cpp 0.4 REQUIRED)
message (" - json-rpc-cpp header: ${JSON_RPC_CPP_INCLUDE_DIRS}")
message (" - json-rpc-cpp lib   : ${JSON_RPC_CPP_LIBRARIES}")

find_package (CUDA)
if (CUDA_FOUND)
	message(" - CUDA header: ${CUDA_INCLUDE_DIRS}")
	message(" - CUDA lib   : ${CUDA_LIBRARIES}")
endif()

# find location of jsonrpcstub
find_program(ETH_JSON_RPC_STUB jsonrpcstub)
message(" - jsonrpcstub location    : ${ETH_JSON_RPC_STUB}")
