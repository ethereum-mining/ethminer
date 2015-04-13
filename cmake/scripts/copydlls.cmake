# this module expects
# DLLS
# CONF
# DESTINATION

# example usage:
# cmake -DDLL_DEBUG=xd.dll -DDLL_RELEASE=x.dll -DCONFIGURATION=Release -DDESTINATION=dest -P scripts/copydlls.cmake

# this script is created cause we do not know configuration in multiconfiguration generators at cmake configure phase ;)

if ("${CONF}" STREQUAL "Release")
    set(DLL ${DLL_RELEASE})
else () # Debug
    set(DLL ${DLL_DEBUG})
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${DLL}" "${DESTINATION}")

