# Find json-rcp-cpp 
#
# Find the json-rpc-cpp includes and library
# 
# if you nee to add a custom library search path, do it via via CMAKE_FIND_ROOT_PATH 
# 
# This module defines
#  JSON_RCP_CPP_INCLUDE_DIRS, where to find header, etc.
#  JSON_RCP_CPP_LIBRARIES, the libraries needed to use json-rpc-cpp.
#  JSON_RCP_CPP_FOUND, If false, do not try to use json-rpc-cpp.

# only look in default directories
find_path(
    JSON_RPC_CPP_INCLUDE_DIR 
    NAMES jsonrpccpp/server.h
    PATH_SUFFIXES jsonrpc
    DOC "json-rpc-cpp include dir"
)

find_library(
    JSON_RPC_CPP_COMMON_LIBRARY
    NAMES jsonrpccpp-common
    DOC "json-rpc-cpp common library"
)

find_library(
    JSON_RPC_CPP_SERVER_LIBRARY
    NAMES jsonrpccpp-server
    DOC "json-rpc-cpp server library"
)

find_library(
    JSON_RPC_CPP_CLIENT_LIBRARY
    NAMES jsonrpccpp-client
    DOC "json-rpc-cpp client library"
)

# message (" - json-rcp-cpp header : ${JSON_RPC_CPP_INCLUDE_DIRS}")
# message (" - json-rcp-cpp lib    : ${JSON_RPC_CPP_LIBRARIES}")


# handle the QUIETLY and REQUIRED arguments and set JSON_RPC_CPP_FOUND to TRUE
# if all listed variables are TRUE, hide their existence from configuration view
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json_rpc_cpp DEFAULT_MSG
    JSON_RPC_CPP_COMMON_LIBRARY JSON_RPC_CPP_SERVER_LIBRARY JSON_RPC_CPP_CLIENT_LIBRARY JSON_RPC_CPP_INCLUDE_DIR)                             
mark_as_advanced (JSON_RPC_CPP_COMMON_LIBRARY JSON_RPC_CPP_SERVER_LIBRARY JSON_RPC_CPP_CLIENT_LIBRARY JSON_RPC_CPP_INCLUDE_DIR)

# these are the variables to be uses by the calling script
set (JSON_RPC_CPP_INCLUDE_DIRS ${JSON_RPC_CPP_INCLUDE_DIR})
set (JSON_RPC_CPP_LIBRARIES ${JSON_RPC_CPP_COMMON_LIBRARY} ${JSON_RPC_CPP_SERVER_LIBRARY} ${JSON_RPC_CPP_CLIENT_LIBRARY})

# message (" - json-rcp-cpp header : ${JSON_RPC_CPP_INCLUDE_DIRS}")
# message (" - json-rcp-cpp lib    : ${JSON_RPC_CPP_LIBRARIES}")
