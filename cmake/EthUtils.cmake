#
# renames the file if it is different from its destination
include(CMakeParseArguments)
#
macro(replace_if_different SOURCE DST)
	set(extra_macro_args ${ARGN})
	set(options CREATE)
	set(one_value_args)
	set(multi_value_args)
	cmake_parse_arguments(REPLACE_IF_DIFFERENT "${options}" "${one_value_args}" "${multi_value_args}" "${extra_macro_args}")

	if (REPLACE_IF_DIFFERENT_CREATE AND (NOT (EXISTS "${DST}")))
		file(WRITE "${DST}" "")
	endif()

	execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${SOURCE}" "${DST}" RESULT_VARIABLE DIFFERENT)

	if (DIFFERENT)
		execute_process(COMMAND ${CMAKE_COMMAND} -E rename "${SOURCE}" "${DST}")
	else()
		execute_process(COMMAND ${CMAKE_COMMAND} -E remove "${SOURCE}")
	endif()
endmacro()

macro(eth_add_test NAME) 

	# parse arguments here
	set(commands)
	set(current_command "")
	foreach (arg ${ARGN})
		if (arg STREQUAL "ARGS")
			if (current_command)
				list(APPEND commands ${current_command})
			endif()
			set(current_command "")
		else ()
			set(current_command "${current_command} ${arg}")
		endif()
	endforeach(arg)
	list(APPEND commands ${current_command})

	message(STATUS "test: ${NAME} | ${commands}")

	# create tests
	set(index 0)
	list(LENGTH commands count)
	while (index LESS count)
		list(GET commands ${index} test_arguments)

		set(run_test "--run_test=${NAME}")
		add_test(NAME "${NAME}.${index}" COMMAND testeth ${run_test} ${test_arguments})
		
		math(EXPR index "${index} + 1")
	endwhile(index LESS count)

	# add target to run them
	add_custom_target("test.${NAME}"
		DEPENDS testeth
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMAND ${CMAKE_COMMAND} -DETH_TEST_NAME="${NAME}" -DCTEST_COMMAND="${CTEST_COMMAND}" -P "${ETH_SCRIPTS_DIR}/runtest.cmake"
	)

endmacro()

# Based on
# http://stackoverflow.com/questions/11813271/embed-resources-eg-shader-code-images-into-executable-library-with-cmake
# Creates C resources file from files
function(eth_create_resources bins output)
	set(tmp_output "${output}.tmp")
	# Create empty output file
	file(WRITE ${tmp_output} "")
	# Collect input files
#	file(GLOB bins ${dir}/*)
	# Iterate through input files
	foreach(bin ${${bins}})
		# Get short filename
		string(REGEX MATCH "([^/]+)$" filename ${bin})
		# Replace filename spaces & extension separator for C compatibility
		string(REGEX REPLACE "\\.| " "_" filename ${filename})
		# Add eth prefix (qt does the same thing)
		set(filename "eth_${filename}")
		# full name
		file(GLOB the_name ${bin})
		# Read hex data from file
		file(READ ${bin} filedata HEX)
		# Convert hex data for C compatibility
		string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
		# Append data to output file
		file(APPEND ${tmp_output} "static const unsigned char ${filename}[] = {\n  // ${the_name}\n  ${filedata}};\nstatic const unsigned ${filename}_size = sizeof(${filename});\n")
	endforeach()
	replace_if_different("${tmp_output}" "${output}")
endfunction()
