# Find readline
#
# Find the readline includes and library
# 
# if you nee to add a custom library search path, do it via via CMAKE_FIND_ROOT_PATH 
# 
# This module defines
#  READLINE_INCLUDE_DIR, where to find header, etc.
#  READLINE_LIBRARY, the libraries needed to use readline.
#  READLINE_FOUND, If false, do not try to use readline.

# only look in default directories
find_path(
	READLINE_INCLUDE_DIR 
	NAMES readline/readline.h
	DOC "readling include dir"
	)

find_library(
	READLINE_LIBRARY
	NAMES readline
	DOC "readline library"
	)

# handle the QUIETLY and REQUIRED arguments and set READLINE_FOUND to TRUE
# if all listed variables are TRUE, hide their existence from configuration view
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(readline DEFAULT_MSG
	READLINE_INCLUDE_DIR READLINE_LIBRARY)
mark_as_advanced (READLINE_INCLUDE_DIR READLINE_LIBRARY)

