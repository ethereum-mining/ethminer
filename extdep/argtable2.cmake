if(APPLE)

elseif(WIN32)
ExternalProject_Add(argtable2
    URL http://sourceforge.net/projects/argtable/files/argtable/argtable-2.13/argtable2-13.tar.gz
    BINARY_DIR argtable2-prefix/src/argtable2
    CONFIGURE_COMMAND cmake .
    BUILD_COMMAND devenv argtable2.sln /build release
    INSTALL_COMMAND cmd /c cp src/Release/argtable2.lib ${ETH_DEPENDENCY_INSTALL_DIR}/lib && cp src/argtable2.h ${ETH_DEPENDENCY_INSTALL_DIR}/include
)
else()
endif()
