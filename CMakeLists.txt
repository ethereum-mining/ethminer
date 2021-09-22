cmake_minimum_required(VERSION 3.5)

include(cmake/cable/bootstrap.cmake)

include(CableBuildInfo)
include(CableBuildType)
include(CableToolchains)
include(HunterGate)

include(defaults/HunterCacheServers)

cable_configure_toolchain(DEFAULT cxx11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(HUNTER_CONFIGURATION_TYPES Release CACHE STRING "Build type of Hunter packages")
set(HUNTER_JOBS_NUMBER 6 CACHE STRING "Number of parallel builds used by Hunter")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.112.tar.gz"
    SHA1 "4b894e1d5d203f0cc9a77431dbb1b486ab6f4430"
    LOCAL
)

project(ethminer)
set(PROJECT_VERSION 0.19.0)

cable_set_build_type(DEFAULT Release CONFIGURATION_TYPES Release RelWithDebInfo)

option(ETHASHCL "Build with OpenCL mining" ON)
option(ETHASHOCL "Build with OpenCL FPGA mining" ON)
option(ETHASHCUDA "Build with CUDA mining" ON)
option(ETHASHCPU "Build with CPU mining (only for development)" OFF)
option(ETHDBUS "Build with D-Bus support" OFF)
option(APICORE "Build with API Server support" ON)
option(BINKERN "Install AMD binary kernels" ON)
option(DEVBUILD "Log developer metrics" OFF)
option(USE_SYS_OPENCL "Build with system OpenCL" OFF)

# propagates CMake configuration options to the compiler
function(configureProject)
    if (ETHASHCL)
        add_definitions(-DETH_ETHASHCL)
<<<<<<< HEAD
=======
      endif()
    if (ETHASHOCL)
        add_definitions(-DETH_ETHASHOCL)
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
    endif()
    if (ETHASHCUDA)
        add_definitions(-DETH_ETHASHCUDA)
    endif()
<<<<<<< HEAD
    if (ETHASHCPU)
        add_definitions(-DETH_ETHASHCPU)
    endif()
=======
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
    if (ETHDBUS)
        add_definitions(-DETH_DBUS)
    endif()
    if (APICORE)
        add_definitions(-DAPI_CORE)
    endif()
<<<<<<< HEAD
    if (BINKERN)
        add_definitions(-DBIN_KERN)
    endif()
    if (DEVBUILD)
        add_definitions(-DDEV_BUILD)
    endif()
    if (USE_SYS_OPENCL)
        add_definitions(-DUSE_SYS_OPENCL)
    endif()
=======
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
endfunction()

hunter_add_package(Boost COMPONENTS system filesystem thread)
find_package(Boost CONFIG REQUIRED COMPONENTS system filesystem thread)

hunter_add_package(jsoncpp)
find_package(jsoncpp CONFIG REQUIRED)

hunter_add_package(ethash)
find_package(ethash CONFIG REQUIRED)

configureProject()

if(APPLE)
    set(USE_SYS_OPENCL ON)
endif()

message("----------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
<<<<<<< HEAD
message("----------------------------------------------------------------- components")
message("-- ETHASHCL         Build OpenCL components                      ${ETHASHCL}")
message("-- ETHASHCUDA       Build CUDA components                        ${ETHASHCUDA}")
message("-- ETHASHCPU        Build CPU components (only for development)  ${ETHASHCPU}")
message("-- ETHDBUS          Build D-Bus components                       ${ETHDBUS}")
message("-- APICORE          Build API Server components                  ${APICORE}")
message("-- BINKERN          Install AMD binary kernels                   ${BINKERN}")
message("-- DEVBUILD         Build with dev logging                       ${DEVBUILD}")
message("-- USE_SYS_OPENCL   Build with system OpenCL                     ${USE_SYS_OPENCL}")
message("----------------------------------------------------------------------------")
=======
message("------------------------------------------------------------- components")
message("-- ETHASHCL         Build OpenCL components                  ${ETHASHCL}")
message("-- ETHASHOCL        Build OpenCL FPGA components             ${ETHASHOCL}")
message("-- ETHASHCUDA       Build CUDA components                    ${ETHASHCUDA}")
message("-- ETHDBUS          Build D-Bus components                   ${ETHDBUS}")
message("-- APICORE          Build API Server components              ${APICORE}")
message("------------------------------------------------------------------------")
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
message("")

include(EthCompilerSettings)
if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")
endif()

cable_add_buildinfo_library(PROJECT_NAME ${PROJECT_NAME})

execute_process(COMMAND aocl compile-config
  OUTPUT_VARIABLE AOCL_COMPILE_CONFIG)
execute_process(COMMAND aocl link-config
  OUTPUT_VARIABLE AOCL_LINK_CONFIG)
string(REGEX REPLACE "\n$" "" AOCL_COMPILE_CONFIG "${AOCL_COMPILE_CONFIG}")
string(REGEX REPLACE "\n$" "" AOCL_LINK_CONFIG "${AOCL_LINK_CONFIG}")

string(REGEX MATCHALL "-l([^ ]*)" AOCL_LIBRARIES "${AOCL_LINK_CONFIG}")
string(REGEX MATCHALL "-L([^ ]*)" AOCL_LINK_SEARCH "${AOCL_LINK_CONFIG}")
string(REGEX REPLACE "-l([^;]*)" "\\1" AOCL_LIBRARIES "${AOCL_LIBRARIES}")
string(REGEX REPLACE "-L([^;]*)" "\\1" AOCL_LINK_SEARCH "${AOCL_LINK_SEARCH}")

separate_arguments(AOCL_COMPILER_OPTS UNIX_COMMAND
  "${AOCL_LINK_CONFIG} ${AOCL_COMPILE_CONFIG}")

# Generic, no need to modify
add_subdirectory(libdevcore)

# Done, only one variable needed to be added on Miner.h
add_subdirectory(libethcore)

# No relevant changes
add_subdirectory(libhwmon)

# No relevant changes
add_subdirectory(libpoolprotocols)

if (ETHASHCL)
    add_subdirectory(libethash-cl)
<<<<<<< HEAD
    if (BINKERN)
        add_subdirectory(libethash-cl/kernels)
    endif()
endif ()
if (ETHASHCUDA)
    add_subdirectory(libethash-cuda)
endif ()
if (ETHASHCPU)
    add_subdirectory(libethash-cpu)
=======
endif ()
if (ETHASHOCL)
    add_subdirectory(libethash-ocl)
endif ()
if (ETHASHCUDA)
    add_subdirectory(libethash-cuda)
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
endif ()
if (APICORE)
    add_subdirectory(libapicore)
endif()

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
<<<<<<< HEAD
=======


# Export some variables for CIs
if(PROJECT_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
    set(PROJECT_VERSION_IS_PRERELEASE false)
else()
    set(PROJECT_VERSION_IS_PRERELEASE true)
endif()
configure_file(cmake/variables.sh.in variables.sh)
configure_file(cmake/variables.ps1.in variables.ps1)
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
