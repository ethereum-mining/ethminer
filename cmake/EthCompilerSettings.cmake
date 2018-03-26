# Set necessary compile and link flags

include(EthCheckCXXFlags)

# C++11 check and activation
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra -Wno-error=parentheses -pedantic")

    eth_add_cxx_compiler_flag_if_supported(-ffunction-sections)
    eth_add_cxx_compiler_flag_if_supported(-fdata-sections)
    eth_add_cxx_linker_flag_if_supported(-Wl,--gc-sections)

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")

    eth_add_cxx_compiler_flag_if_supported(-ffunction-sections)
    eth_add_cxx_compiler_flag_if_supported(-fdata-sections)
    eth_add_cxx_linker_flag_if_supported(-Wl,--gc-sections)

	if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -fcolor-diagnostics -Qunused-arguments")
	endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

	# declare Windows XP requirement
	# undefine windows.h MAX & MIN macros because they conflict with std::min & std::max functions
	# disable Unsafe CRT Library functions warnings
	add_definitions(/D_WIN32_WINNT=0x0501 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS)

	# specify Exception Handling Model
	# disable conversion from "type1" to "type2", possible loss of data (C4244)
	# disable conversion from "size_t" to "type", possible loss of data (C4267)
	# disable C++ exception specification ignored except to indicate a function is not __declspec(nothrow) (C4290)
	add_compile_options(/EHsc /wd4244 /wd4267 /wd4290)

	# Release/RelWithDebInfo builds
	# enable parallel compilation
	# enable LTCG for faster builds
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MP /GL")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE}")
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
	set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE}")

	# enable LTCG for faster builds
	set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
	set(CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_STATIC_LINKER_FLAGS_RELEASE}")

	# disable incremental linking
	# enable LTCG for faster builds
	# enable unused references removal
	# enable RELEASE so that the executable file has its checksum set
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO /LTCG /OPT:REF /OPT:ICF /RELEASE")
	set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")

else ()
	message(WARNING "Your compiler is not tested, if you run into any issues, we'd welcome any patches.")
endif ()

set(SANITIZE NO CACHE STRING "Instrument build with provided sanitizer")
if(SANITIZE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${SANITIZE}")
endif()
