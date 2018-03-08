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
else ()
	message(WARNING "Your compiler is not tested, if you run into any issues, we'd welcome any patches.")
endif ()

set(SANITIZE NO CACHE STRING "Instrument build with provided sanitizer")
if(SANITIZE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${SANITIZE}")
endif()

