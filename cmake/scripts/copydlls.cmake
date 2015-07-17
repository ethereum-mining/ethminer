# this module expects
# DLLS
# CONF
# DESTINATION

# example usage:
# cmake -DDLL_DEBUG=xd.dll -DDLL_RELEASE=x.dll -DCONFIGURATION=Release -DDESTINATION=dest -P scripts/copydlls.cmake

# this script is created cause we do not know configuration in multiconfiguration generators at cmake configure phase ;)

if ("${CONF}" STREQUAL "Debug")
    set(DLL ${DLL_DEBUG})
else ()
    set(DLL ${DLL_RELEASE})
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${DLL}" "${DESTINATION}")

