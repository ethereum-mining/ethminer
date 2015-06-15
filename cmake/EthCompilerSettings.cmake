# Set necessary compile and link flags

# C++11 check and activation
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -Werror -pedantic -DSHAREDLIB -fPIC ${CMAKE_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DETH_DEBUG")
	set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG -DETH_RELEASE")
	set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG -DETH_RELEASE")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DETH_RELEASE")

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
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DETH_RELEASE")

	if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -fcolor-diagnostics -Qunused-arguments -DBOOST_ASIO_HAS_CLANG_LIBCXX")
	endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

	# enable parallel compilation
	# specify Exception Handling Model in msvc
	# disable unknown pragma warning (4068)
	# disable unsafe function warning (4996)
	# disable decorated name length exceeded, name was truncated (4503)
	# disable warning C4535: calling _set_se_translator() requires /EHa (for boost tests)
	# declare Windows XP requirement
	# undefine windows.h MAX && MIN macros cause it cause conflicts with std::min && std::max functions
	# define miniupnp static library
	add_compile_options(/MP /EHsc /wd4068 /wd4996 /wd4503 -D_WIN32_WINNT=0x0501 /DNOMINMAX /DMINIUPNP_STATICLIB)
	# disable empty object file warning
	set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /ignore:4221")
	# warning LNK4075: ignoring '/EDITANDCONTINUE' due to '/SAFESEH' specification 
	# warning LNK4099: pdb was not found with lib
	# stack size 16MB
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4099,4075 /STACK:33554432")

	# windows likes static
	if (NOT ETH_STATIC)
		message("Forcing static linkage for MSVC.")
		set(ETH_STATIC 1)
	endif ()
else ()
	message(WARNING "Your compiler is not tested, if you run into any issues, we'd welcome any patches.")
endif ()

if (PROFILING AND (("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")))
	set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")
	set(CMAKE_C_FLAGS "-g ${CMAKE_C_FLAGS}")
	add_definitions(-DETH_PROFILING_GPERF)
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lprofiler")
#	set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -lprofiler")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lprofiler")
endif ()

if (("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
	option(USE_LD_GOLD "Use GNU gold linker" ON)
	if (USE_LD_GOLD)
		execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version ERROR_QUIET OUTPUT_VARIABLE LD_VERSION)
		if ("${LD_VERSION}" MATCHES "GNU gold")
			set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
		endif ()
	endif ()
endif ()

if(ETH_STATIC)
	set(BUILD_SHARED_LIBS OFF)
else()
	set(BUILD_SHARED_LIBS ON)
endif(ETH_STATIC)
