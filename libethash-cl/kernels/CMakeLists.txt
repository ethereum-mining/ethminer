include(GNUInstallDirs)
file (GLOB BINS "bin/*.bin")
install(FILES ${BINS} DESTINATION ${CMAKE_INSTALL_BINDIR}/kernels)
