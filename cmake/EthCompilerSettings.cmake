# Set necessary compile and link flags


# C++11 check and activation
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	execute_process(
		COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
	if (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
		message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
	endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
else ()
	message(FATAL_ERROR "Your C++ compiler does not support C++11.")
endif ()



# Initialize CXXFLAGS
set(CMAKE_CXX_FLAGS                "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -DSHAREDLIB")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DETH_DEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG -DETH_RELEASE")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG -DETH_RELEASE")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DETH_DEBUG")

# Windows 
if ("${TARGET_PLATFORM}" STREQUAL "w64")
	set(CMAKE_SYSTEM_NAME Windows)

	set(CMAKE_CXX_LIBRARY_ARCHITECTURE	x86_64-w64-mingw32)
	set(CMAKE_C_COMPILER			x86_64-w64-mingw32-gcc)
	set(CMAKE_CXX_COMPILER			x86_64-w64-mingw32-g++)
	set(CMAKE_RC_COMPILER			x86_64-w64-mingw32-windres)
	set(CMAKE_AR				x86_64-w64-mingw32-ar)
	set(CMAKE_RANLIB			x86_64-w64-mingw32-ranlib)

	set(CMAKE_EXECUTABLE_SUFFIX		.exe)

	set(CMAKE_FIND_ROOT_PATH
		/usr/x86_64-w64-mingw32
	)

	include_directories(/usr/x86_64-w64-mingw32/include/cryptopp)

	set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
	set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
	set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

	set(CMAKE_INSTALL_PREFIX	/usr/x86_64-w64-mingw32)
	set(ETH_BUILD_PLATFORM "windows")
	set(ETH_STATIC 1)
else ()
	set(ETH_BUILD_PLATFORM ${CMAKE_SYSTEM_NAME})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
	set(ETH_SHARED 1)
endif()



