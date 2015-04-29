
# cmake -DETH_RES_FILE=... -DETH_DST_NAME=... -P scripts/resources.cmake
# cmake -DETH_RES_FILE=test.cmake -DETH_DST_NAME=dst -P resources.cmake

# should define ETH_RESOURCES
include(${ETH_RES_FILE})

set(ETH_RESULT_DATA)
set(ETH_RESULT_INIT)

# resource is a name visible for cpp application 
foreach(resource ${ETH_RESOURCES})
	
	# filename is the name of file which will be used in app
	set(filename ${${resource}})

	# filedata is a file content
	file(READ ${filename} filedata HEX)

	# Convert hex data for C compatibility
	string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})

	# append static variables to result variable
	list(APPEND ${ETH_RESULT_DATA} "static const unsigned char eth_${resource}[] = {\n  // ${filename}\n  ${filedata}\n};\n")

	# append init resources
	list(APPEND ${ETH_RESULT_INIT} "	eth_resources[\"${resource}\"] = (char const*)eth_${resource};\n")
	list(APPEND ${ETH_RESULT_INIT} "	eth_sizes[\"${resource}\"]     = sizeof(eth_${resource});\n")

endforeach(resource)

configure_file("resource.cpp.in" "${ETH_DST_NAME}.cpp.tmp")

include("../EthUtils.cmake")
replace_if_different("${ETH_DST_NAME}.cpp.tmp" "${ETH_DST_NAME}.cpp")
replace_if_different("resource.h" "${ETH_DST_NAME}.h")

