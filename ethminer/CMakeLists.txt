cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)

set(EXECUTABLE ethminer)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

hunter_add_package(CLI11)
find_package(CLI11 CONFIG REQUIRED)

target_link_libraries(ethminer PRIVATE ethcore poolprotocols devcore ethminer-buildinfo CLI11::CLI11 Boost::system)

if(APICORE)
	target_link_libraries(ethminer PRIVATE apicore)
endif()

include(GNUInstallDirs)
install(TARGETS ethminer DESTINATION ${CMAKE_INSTALL_BINDIR})
if(MSVC)
    install(FILES $<TARGET_PDB_FILE:ethminer> DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
endif()
