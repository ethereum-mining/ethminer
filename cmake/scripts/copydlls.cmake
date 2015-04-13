# this module expects
# DLLS
# CONFIGURATION
# DESTINATION

# example usage:
# cmake -DLLS=${MHD_LIBRARIES} -DCONFIGURATION=Release -DDESTINATION=dest -P scripts/copydlls.cmake

# expects DLLS to be in format optimized;path_to_dll.dll;debug;path_to_dll_d.dll
if (${CONFIGURATION} STREQUAL "Release")
    list(GET DLLS 1 DLL)
else () # Debug
    list(GET DLLS 3 DLL)
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${DLL}" "${DESTINATION}")

