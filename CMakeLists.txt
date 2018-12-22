cmake_minimum_required(VERSION 3.5)

include(cmake/cable/bootstrap.cmake)

include(CableBuildInfo)
include(CableBuildType)
include(CableToolchains)
include(HunterGate)

include(defaults/HunterCacheServers)

cable_configure_toolchain(DEFAULT cxx11)

set(HUNTER_CONFIGURATION_TYPES Release CACHE STRING "Build type of Hunter packages")
set(HUNTER_JOBS_NUMBER 6 CACHE STRING "Number of parallel builds used by Hunter")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.6.tar.gz"
    SHA1 "951e8daf57a51708b0e6a00cab342a042db57a2f"
    LOCAL
)

project(ethminer)
set(PROJECT_VERSION 0.18.0-alpha.3)

cable_set_build_type(DEFAULT Release CONFIGURATION_TYPES Release RelWithDebInfo)


option(OPENCL "Build with OpenCL mining" ON)
option(CUDA "Build with CUDA mining" ON)
option(DBUS "Build with D-Bus support" OFF)
option(API "Build with API Server support" ON)
option(CPU "Build with CPU mining (only for development)" OFF)
option(BINKERN "Install AMD binary kernels" ON)
option(DEVBUILD "Log developer metrics" OFF)



# propagates CMake configuration options to the compiler
function(configureProject)
	if (OPENCL)
		add_definitions(-D_OPENCL)
	else
	    set(BINKERN OFF)
	endif()
	if (CUDA)
		add_definitions(-D_CUDA)
	endif()
	if (DBUS)
		add_definitions(-D_DBUS)
    endif()
	if (CPU)
		add_definitions(-D_CPU)
	endif()
	if (API)
		add_definitions(-D_API)
	endif()
	if (BINKERN)
		add_definitions(-D_BINKERN)
	endif()
	if (DEVBUILD)
		add_definitions(-D_DEVELOPER)
	endif()
endfunction()

hunter_add_package(Boost COMPONENTS system filesystem thread)
find_package(Boost CONFIG REQUIRED COMPONENTS system filesystem thread)

hunter_add_package(jsoncpp)
find_package(jsoncpp CONFIG REQUIRED)

hunter_add_package(ethash)
find_package(ethash CONFIG REQUIRED)

configureProject()

message("--------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")

message("--------------------------------------------------------------------------")
message("-- OPENCL         Build OpenCL mining components               ${OPENCL}")
message("-- CUDA           Build CUDA mining components                 ${CUDA}")
message("-- CPU            Build CPU mining components                  ${CPU}")
message("-- DBUS           Build D-Bus components                       ${DBUS}")
message("-- API            Build API Server components                  ${API}")
message("-- BINKERN        Enable AMD OpenCL binary mining kernels      ${BINKERN}")
message("-- DEVBUILD       Build with developer logging                 ${DEVBUILD}")
message("--------------------------------------------------------------------------")
message("")

include(EthCompilerSettings)
if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")
endif()


cable_add_buildinfo_library(PROJECT_NAME ${PROJECT_NAME})

add_subdirectory(libprogpow)
add_subdirectory(libdevcore)
add_subdirectory(libethcore)
add_subdirectory(libhwmon)
add_subdirectory(libpoolprotocols)

if (OPENCL)
	add_subdirectory(libethash-cl)
	if (BINKERN)
		add_subdirectory(libethash-cl/kernels)
	endif()
endif ()
if (CUDA)
	add_subdirectory(libethash-cuda)
endif ()
if (API)
	add_subdirectory(libapicore)
endif()
if (CPU)
	add_subdirectory(libethash-cpu)
endif ()

add_subdirectory(ethminer)


if(WIN32)
	set(CPACK_GENERATOR ZIP)
else()
	set(CPACK_GENERATOR TGZ)
endif()
set(CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY FALSE)
include(CPack)
