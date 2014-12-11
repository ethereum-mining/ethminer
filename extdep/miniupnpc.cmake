# TODO this file is not used yet, but will be in future
include(ExternalProject)

ExternalProject_Add(miniupnpc
	URL http://miniupnp.tuxfamily.org/files/download.php?file=miniupnpc-1.9.20141027.tar.gz 
	BINARY_DIR miniupnpc-prefix/src/miniupnpc
	CONFIGURE_COMMAND ""
	BUILD_COMMAND make -j 3
	INSTALL_COMMAND make install INSTALLPREFIX=${ETH_DEPENDENCY_INSTALL_DIR}
)

