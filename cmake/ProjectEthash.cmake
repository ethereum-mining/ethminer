include(ExternalProject)
include(GNUInstallDirs)

set(prefix ${CMAKE_BINARY_DIR}/deps)
set(ethash_include_dir ${prefix}/${CMAKE_INSTALL_INCLUDEDIR})
set(ethash_library ${prefix}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ethash${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Add(
    ethash-project
    PREFIX ${prefix}
    GIT_REPOSITORY https://github.com/chfast/ethash
    GIT_TAG 701c4c94fbaf525c4e5e871fe466a90b1aa6228e
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DETHASH_BUILD_TESTS=OFF
    -DETHASH_INSTALL_CMAKE_CONFIG=OFF
    -DHUNTER_ENABLED=OFF
)

add_library(ethash::ethash STATIC IMPORTED)
file(MAKE_DIRECTORY ${ethash_include_dir})
set_target_properties(
    ethash::ethash
    PROPERTIES
    IMPORTED_LOCATION ${ethash_library}
    INTERFACE_INCLUDE_DIRECTORIES ${ethash_include_dir}
)
add_dependencies(ethash::ethash ethash-project)
