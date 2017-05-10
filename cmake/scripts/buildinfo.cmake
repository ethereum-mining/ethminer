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

if (NOT ETH_BUILD_TYPE)
	set(ETH_BUILD_TYPE "unknown")
endif()

if (NOT ETH_BUILD_PLATFORM)
	set(ETH_BUILD_PLATFORM "unknown")
endif()

set(INFILE "${ETH_SOURCE_DIR}/BuildInfo.h.in")
set(TMPFILE "${ETH_DST_DIR}/BuildInfo.h.tmp")
set(OUTFILE "${ETH_DST_DIR}/BuildInfo.h")

configure_file("${INFILE}" "${TMPFILE}")

include("${ETH_SOURCE_DIR}/cmake/EthUtils.cmake")
replace_if_different("${TMPFILE}" "${OUTFILE}" CREATE)
