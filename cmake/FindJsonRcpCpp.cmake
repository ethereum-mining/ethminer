# Find json-rcp-cpp 
#
# Find the JSONCpp includes and library
# 
# if you nee to add a custom library search path, do it via via CMAKE_FIND_ROOT_PATH 
# 
# This module defines
#  JSON_RCP_CPP_INCLUDE_DIRS, where to find json.h, etc.
#  JSON_RCP_CPP_LIBRARIES, the libraries needed to use jsoncpp.
#  JSON_RCP_CPP_FOUND, If false, do not try to use jsoncpp.

if (JSON_RPC_CPP_LIBRARIES AND JSON_RPC_CPP_INCLUDE_DIRS)
  # in cache already
  set(JSON_RPC_CPP_FOUND TRUE)
endif()


# only look in default directories
find_path(
    JSON_RPC_CPP_INCLUDE_DIR 
    NAMES jsonrpc/rpc.h
    PATH_SUFFIXES jsonrpc
    DOC "json-rpc-cpp include dir"
)

find_library(
    JSON_RPC_CPP_LIBRARY
    NAMES jsonrpc
    DOC "json-rpc-cpp library"
)

set (JSON_RPC_CPP_INCLUDE_DIRS ${JSON_RPC_CPP_INCLUDE_DIR})
set (JSON_RPC_CPP_LIBRARIES ${JSON_RPC_CPP_LIBRARY})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set JSON_RPC_CPP_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(json-rpc-cpp DEFAULT_MSG
     JSON_RPC_CPP_LIBRARY JSON_RPC_CPP_INCLUDE_DIR)                             


# include(FindPackageMessage)
# find_package_message ("json-rpc-cpp" "found" "bla")


mark_as_advanced (JSON_RPC_CPP_INCLUDE_DIR JSON_RPC_CPP_LIBRARY)                             
