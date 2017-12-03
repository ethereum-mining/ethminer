cmake_minimum_required(VERSION 3.3)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/cmake/toolchain.cmake CACHE FILEPATH "CMake toolchain file")

set(CMAKE_CONFIGURATION_TYPES Release RelWithDebInfo Debug)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type on single-configuration generators")

set(HUNTER_CONFIGURATION_TYPES Release)
set(HUNTER_CACHE_SERVERS "https://github.com/ethereum/hunter-cache")
include(HunterGate)
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.19.86.tar.gz"
    SHA1 "aafb99b93d466f963d53a39271f20d5074996cb0"
	LOCAL
)

if(HUNTER_ENABLED)
	find_package(PythonInterp)
	if(PYTHONINTERP_FOUND)
		hunter_gate_self(
			"${HUNTER_CACHED_ROOT}"
			"${HUNTER_VERSION}"
			"${HUNTER_SHA1}"
			hunter_dir
		)
		set(hunter_upload_script "${hunter_dir}/maintenance/upload-cache-to-github.py")
		set(hunter_cache_dir "${HUNTER_GATE_ROOT}/_Base/Cache")
		set(hunter_tmp_dir "${HUNTER_GATE_ROOT}/tmp")
		add_custom_target(
			hunter_upload_cache
			${PYTHON_EXECUTABLE} ${hunter_upload_script}
			--username hunter-cache-bot
			--repo-owner ethereum
			--repo hunter-cache
			--cache-dir ${hunter_cache_dir}
			--temp-dir ${hunter_tmp_dir}
		)
	endif()
endif()


project(ethminer)
set(PROJECT_VERSION 0.13.0.dev0)

# link_directories interprate relative paths with respect to CMAKE_CURRENT_SOURCE_DIR
cmake_policy(SET CMP0015 NEW)

# let cmake autolink dependencies on windows
# it's specified globally, cause qt libraries requires that on windows and they are also found globally
cmake_policy(SET CMP0020 NEW)

# 3.1 and above
cmake_policy(SET CMP0054 NEW)

######################################################################################################

option(ETHASHCL "Build with OpenCL mining" ON)
option(ETHASHCUDA "Build with CUDA mining" OFF)
option(ETHSTRATUM "Build with Stratum protocol support" ON)
option(ETHDBUS "Build with D-Bus support" OFF)
option(APICORE "Build with API Server support" ON)

# propagates CMake configuration options to the compiler
function(configureProject)
	if (ETHASHCL)
		add_definitions(-DETH_ETHASHCL)
	endif()
	if (ETHASHCUDA)
		add_definitions(-DETH_ETHASHCUDA)
	endif()
	if (ETHSTRATUM)
		add_definitions(-DETH_STRATUM)
	endif()
	if (ETHDBUS)
		add_definitions(-DETH_DBUS)
	endif()
	if (APICORE)
		add_definitions(-DAPI_CORE)
	endif()
endfunction()

function(createBuildInfo)
	# Set build platform; to be written to BuildInfo.h
	set(ETH_BUILD_PLATFORM "${CMAKE_SYSTEM_NAME}")
	if (CMAKE_COMPILER_IS_MINGW)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/mingw")
	elseif (CMAKE_COMPILER_IS_MSYS)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/msys")
	elseif (CMAKE_COMPILER_IS_GNUCXX)
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/g++")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/msvc")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/clang")
	else ()
		set(ETH_BUILD_PLATFORM "${ETH_BUILD_PLATFORM}/unknown")
	endif ()

	#cmake build type may be not specified when using msvc
	if (CMAKE_BUILD_TYPE)
		set(_cmake_build_type ${CMAKE_BUILD_TYPE})
	else()
		set(_cmake_build_type "${CMAKE_CFG_INTDIR}")
	endif()

	# Generate header file containing useful build information
	add_custom_target(BuildInfo.h ALL
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMAND ${CMAKE_COMMAND} -DETH_SOURCE_DIR="${CMAKE_SOURCE_DIR}" -DETH_DST_DIR="${CMAKE_BINARY_DIR}"
			-DETH_BUILD_TYPE="${_cmake_build_type}" -DETH_BUILD_PLATFORM="${ETH_BUILD_PLATFORM}"
			-DPROJECT_VERSION="${PROJECT_VERSION}"
			-P cmake/scripts/buildinfo.cmake
		)
	include_directories(${CMAKE_CURRENT_BINARY_DIR})
endfunction()


hunter_add_package(Boost COMPONENTS system)
find_package(Boost CONFIG REQUIRED COMPONENTS system)

hunter_add_package(jsoncpp)
find_package(jsoncpp CONFIG REQUIRED)

hunter_add_package(libjson-rpc-cpp)
find_package(libjson-rpc-cpp CONFIG REQUIRED)

configureProject()

message("------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
message("------------------------------------------------------------- components")
message("-- ETHASHCL         Build OpenCL components                  ${ETHASHCL}")
message("-- ETHASHCUDA       Build CUDA components                    ${ETHASHCUDA}")
message("-- ETHSTRATUM       Build Stratum components                 ${ETHSTRATUM}")
message("-- ETHDBUS          Build D-Bus components                   ${ETHDBUS}")
message("-- APICORE          Build API Server components              ${APICORE}")
message("------------------------------------------------------------------------")
message("")

include(EthCompilerSettings)

createBuildInfo()

add_subdirectory(libdevcore)
add_subdirectory(libethash)
add_subdirectory(libhwmon)
if (ETHASHCL)
	add_subdirectory(libethash-cl)
endif ()
if (ETHASHCUDA)
	add_subdirectory(libethash-cuda)
endif ()
if(ETHSTRATUM)
	add_subdirectory(libstratum)
endif()
if (APICORE)
	add_subdirectory(libapicore)
endif()
add_subdirectory(libethcore)
add_subdirectory(ethminer)


if(WIN32)
	set(CPACK_GENERATOR ZIP)
else()
	set(CPACK_GENERATOR TGZ)
endif()
set(CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME})
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY FALSE)
include(CPack)


# Export some variables for CIs
if(PROJECT_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
	set(PROJECT_VERSION_IS_PRERELEASE false)
else()
	set(PROJECT_VERSION_IS_PRERELEASE true)
endif()
configure_file(cmake/variables.sh.in variables.sh)
configure_file(cmake/variables.ps1.in variables.ps1)
