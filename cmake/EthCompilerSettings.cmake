# Set necessary compile and link flags

# C++11 check and activation
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -DSHAREDLIB -fPIC")
	set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DETH_DEBUG")
	set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG -DETH_RELEASE")
	set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG -DETH_RELEASE")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DETH_DEBUG")
	set(ETH_SHARED 1)

	execute_process(
		COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
	if (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
		message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
	endif ()

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -DSHAREDLIB -fPIC")
	set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DETH_DEBUG")
	set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG -DETH_RELEASE")
	set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG -DETH_RELEASE")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DETH_DEBUG")
	set(ETH_SHARED 1)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

	# specify Exception Handling Model in msvc
	set(CMAKE_C_FLAGS "/EHsc")
	set(CMAKE_CXX_FLAGS "/EHsc")
	# windows likes static
	set(ETH_STATIC 1)

else ()
	message(WARNING "Your compiler is not tested, if you run into any issues, we'd welcome any patches.")
endif ()

