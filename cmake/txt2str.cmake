# https://gist.github.com/sivachandran/3a0de157dccef822a230
include(CMakeParseArguments)

# Script to wrap opencl text with raw string delimiters and declare static char pointer. 
# Parameters
#   SOURCE_FILE     - The path of source file whose contents will be embedded in the header file.
#   VARIABLE_NAME   - The name of the variable for the string constant.
#   HEADER_FILE     - The path of header file.

set(oneValueArgs SOURCE_FILE VARIABLE_NAME HEADER_FILE)

# reads source file contents as hex string
file(READ ${TXT2STR_SOURCE_FILE} asciiString)

# wrte the wrapped string declaration
file(WRITE ${TXT2STR_HEADER_FILE}
"static const char* ${TXT2STR_VARIABLE_NAME} = R\"delim(\n\n${asciiString}\n\n)delim\";\n")

