# Find ethereum
#
# Find the ethereum includes and library
# 
# if you nee to add a custom library search path, do it via via CMAKE_PREFIX_PATH 
# 
# This module defines
#  ETH_INCLUDE_DIRS, where to find header, etc.
#  ETH_CORE_LIBRARIES, the libraries needed to use ethereum.
#  ETH_FOUND, If false, do not try to use ethereum.

# only look in default directories
find_path(
	ETH_INCLUDE_DIR 
	NAMES ethereum.h
	DOC "ethereum include dir"
)

set(CORE_LIBS web3jsonrpc;webthree;whisper;ethereum;evm;ethcore;lll;p2p;evmasm;devcrypto;evmcore;natspec;devcore;ethash-cl;ethash;secp256k1;scrypt;jsqrc)
set(ALL_LIBS ${CORE_LIBS};evmjit;solidity;secp256k1)

set(ETH_INCLUDE_DIRS ${ETH_INCLUDE_DIR})
set(ETH_CORE_LIBRARIES ${ETH_LIBRARY})

foreach (l ${ALL_LIBS})
	string(TOUPPER ${l} L)
	find_library(ETH_${L}_LIBRARY 
		NAMES ${l}
		PATH_SUFFIXES "lib${l}" "${l}"
	)
endforeach()

foreach (l ${CORE_LIBS})
	string(TOUPPER ${l} L)
	list(APPEND ETH_CORE_LIBRARIES ${ETH_${L}_LIBRARY})
endforeach()

# handle the QUIETLY and REQUIRED arguments and set ETH_FOUND to TRUE
# if all listed variables are TRUE, hide their existence from configuration view
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ethereum DEFAULT_MSG
	ETH_CORE_LIBRARIES)
mark_as_advanced (ETH_CORE_LIBRARIES)

