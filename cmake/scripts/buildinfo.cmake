# generates BuildInfo.h
# 
# this module expects
# ETH_SOURCE_DIR - main CMAKE_SOURCE_DIR
# ETH_DST_DIR - main CMAKE_BINARY_DIR
# ETH_BUILD_TYPE
# ETH_BUILD_PLATFORM
#
# example usage:
# cmake -DETH_SOURCE_DIR=. -DETH_DST_DIR=build -DETH_BUILD_TYPE=Debug -DETH_BUILD_PLATFORM=mac -P scripts/buildinfo.cmake

execute_process(
	COMMAND git --git-dir=${ETH_SOURCE_DIR}/.git --work-tree=${ETH_SOURCE_DIR} rev-parse HEAD
	OUTPUT_VARIABLE ETH_COMMIT_HASH
) 

if (NOT ETH_COMMIT_HASH)
    set(ETH_COMMIT_HASH 0)
endif()

execute_process(
	COMMAND git --git-dir=${ETH_SOURCE_DIR}/.git --work-tree=${ETH_SOURCE_DIR} diff --shortstat	
	OUTPUT_VARIABLE ETH_LOCAL_CHANGES
)

if (ETH_LOCAL_CHANGES)
	set(ETH_CLEAN_REPO 0)
else()
	set(ETH_CLEAN_REPO 1)
endif()

configure_file("${ETH_SOURCE_DIR}/BuildInfo.h.in" "${ETH_DST_DIR}/BuildInfo.h")

