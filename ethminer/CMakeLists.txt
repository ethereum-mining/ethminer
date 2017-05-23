cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)

set(EXECUTABLE ethminer)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

add_dependencies(${EXECUTABLE} BuildInfo.h)

target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} ethash)
target_link_libraries(${EXECUTABLE} devcore libjson-rpc-cpp::client)

if(ETHSTRATUM)
    target_link_libraries(${EXECUTABLE} ethstratum)
endif()

include(GNUInstallDirs)
install(TARGETS ethminer DESTINATION ${CMAKE_INSTALL_BINDIR})
if(MSVC)
    install(FILES $<TARGET_PDB_FILE:ethminer> DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
endif()
