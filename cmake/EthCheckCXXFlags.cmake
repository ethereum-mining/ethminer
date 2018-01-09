include(CheckCXXCompilerFlag)

# Adds CXX compiler flag if the flag is supported by the compiler.
#
# This is effectively a combination of CMake's check_cxx_compiler_flag()
# and add_compile_options():
#
#    if(check_cxx_compiler_flag(flag))
#        add_compile_options(flag)
#
function(eth_add_cxx_compiler_flag_if_supported FLAG)
  # Remove leading - or / from the flag name.
  string(REGEX REPLACE "^-|/" "" name ${FLAG})
  check_cxx_compiler_flag(${FLAG} ${name})
  if(${name})
    add_compile_options(${FLAG})
  endif()
  # If the optional argument passed, store the result there.
  if(ARGV1)
    set(${ARGV1} ${name} PARENT_SCOPE)
  endif()
endfunction()

# Adds CXX linker flag if the flag is supported by the linker.
#
# This is effectively a combination of CMake's check_cxx_compiler_flag()
# which runs a test compile and link, with an update of the 
# CMAKE_CXX_LINK_FLAGS which is passed to the CXX driver for
# the link phase.
#
#    if(check_cxx_compiler_flag(flag))
#        CMAKE_CXX_LINK_FLAGS += flag
#
function(eth_add_cxx_linker_flag_if_supported FLAG)
  # Remove leading - or / from the flag name.
  string(REGEX REPLACE "^-|/" "" name ${FLAG})
  check_cxx_compiler_flag(${FLAG} ${name})
  if(${name})
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${FLAG}")
  endif()
  # If the optional argument passed, store the result there.
  if(ARGV1)
    set(${ARGV1} ${name} PARENT_SCOPE)
  endif()
endfunction()
