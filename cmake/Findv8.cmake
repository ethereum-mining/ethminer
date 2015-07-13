# Find v8
#
# Find the v8 includes and library
# 
# if you nee to add a custom library search path, do it via via CMAKE_PREFIX_PATH 
# 
# This module defines
#  V8_INCLUDE_DIRS, where to find header, etc.
#  V8_LIBRARIES, the libraries needed to use v8.
#  V8_FOUND, If false, do not try to use v8.

# only look in default directories
find_path(
	V8_INCLUDE_DIR 
	NAMES v8.h
	DOC "v8 include dir"
)

find_library(
	V8_LIBRARY
	NAMES v8
	DOC "v8 library"
)

set(V8_INCLUDE_DIRS ${V8_INCLUDE_DIR})
set(V8_LIBRARIES ${V8_LIBRARY})

# debug library on windows
# same naming convention as in qt (appending debug library with d)
# boost is using the same "hack" as us with "optimized" and "debug"
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

	find_library(
		V8_LIBRARY
		NAMES v8_base
		DOC "v8 base library"
	)

	find_library(
		V8_NO_SNAPSHOT_LIBRARY
		NAMES v8_nosnapshot
		DOC "v8 nosnapshot library"
	)
	
	set(V8_LIBRARIES ${V8_LIBRARY} ${V8_NO_SNAPSHOT_LIBRARY})

	find_library(
		V8_LIBRARY_DEBUG
		NAMES v8_based
		DOC "v8 base library"
	)

	find_library(
		V8_NO_SNAPSHOT_LIBRARY_DEBUG
		NAMES v8_nosnapshotd
		DOC "v8 nosnapshot library"
	)

	set(V8_LIBRARIES "ws2_32" "winmm" optimized ${V8_LIBRARIES} debug ${V8_LIBRARY_DEBUG} ${V8_NO_SNAPSHOT_LIBRARY_DEBUG})

endif()

# handle the QUIETLY and REQUIRED arguments and set V8_FOUND to TRUE
# if all listed variables are TRUE, hide their existence from configuration view
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(v8 DEFAULT_MSG
	V8_INCLUDE_DIR V8_LIBRARY)
mark_as_advanced (V8_INCLUDE_DIR V8_LIBRARY)

