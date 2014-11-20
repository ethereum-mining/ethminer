# - try to find JSONCPP library
#
# Cache Variables: (probably not for direct use in your scripts)
#  JSONCPP_INCLUDE_DIR
#  JSONCPP_LIBRARY
#
# Non-cache variables you might use in your CMakeLists.txt:
#  JSONCPP_FOUND
#  JSONCPP_INCLUDE_DIRS
#  JSONCPP_LIBRARIES
#
# Requires these CMake modules:
#  FindPackageHandleStandardArgs (known included with CMake >=2.6.2)
#
# Author:
# 2011 Philippe Crassous (ENSAM ParisTech / Institut Image) p.crassous _at_ free.fr
#
# Adapted from the Virtual Reality Peripheral Network library.
# https://github.com/rpavlik/vrpn/blob/master/README.Legal
#

set(JSONCPP_ROOT_DIR
	"${JSONCPP_ROOT_DIR}"
	CACHE
	PATH
	"Directory to search for JSONCPP")
set(_jsoncppnames)
set(_pathsuffixes
	suncc
	vacpp
	mingw
	msvc6
	msvc7
	msvc71
	msvc80
	msvc90
	linux-gcc)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	execute_process(COMMAND
		${CMAKE_CXX_COMPILER}
		-dumpversion
		OUTPUT_VARIABLE
		_gnucxx_ver
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	list(APPEND
		_jsoncppnames
		json_linux-gcc-${_gnucxx_ver}_libmt
		json_linux-gcc_libmt)
	list(APPEND _pathsuffixes linux-gcc-${_gnucxx_ver})
elseif(MSVC)
	if(MSVC_VERSION EQUAL 1200)
		list(APPEND _jsoncppnames json_vc6_libmt)
	elseif(MSVC_VERSION EQUAL 1300)
		list(APPEND _jsoncppnames json_vc7_libmt)
	elseif(MSVC_VERSION EQUAL 1310)
		list(APPEND _jsoncppnames json_vc71_libmt)
	elseif(MSVC_VERSION EQUAL 1400)
		list(APPEND _jsoncppnames json_vc8_libmt)
	elseif(MSVC_VERSION EQUAL 1500)
		list(APPEND _jsoncppnames json_vc9_libmt)
	elseif(MSVC_VERSION EQUAL 1600)
		list(APPEND _jsoncppnames json_vc10_libmt)
	endif()
else()
	list(APPEND _jsoncppnames
		json_suncc_libmt
		json_vacpp_libmt)
endif()

list(APPEND _jsoncppnames
	json_mingw_libmt
    jsoncpp)

find_library(JSONCPP_LIBRARY
	NAMES
	${_jsoncppnames}
	PATHS
	"${JSONCPP_ROOT_DIR}/libs"
	PATH_SUFFIXES
	${_pathsuffixes})

find_path(JSONCPP_INCLUDE_DIR
	NAMES
	json/json.h
	PATHS
	"${JSONCPP_ROOT_DIR}"
	PATH_SUFFIXES jsoncpp
	include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JSONCPP
	DEFAULT_MSG
	JSONCPP_LIBRARY
	JSONCPP_INCLUDE_DIR)

if(JSONCPP_FOUND)
	set(JSONCPP_LIBRARIES "${JSONCPP_LIBRARY}")
	set(JSONCPP_INCLUDE_DIRS "${JSONCPP_INCLUDE_DIR}")
	mark_as_advanced(JSONCPP_ROOT_DIR)
endif()

mark_as_advanced(JSONCPP_INCLUDE_DIR JSONCPP_LIBRARY)

