if (APPLE)

elseif (WIN32)
set(boost_address_model)
# on windows 64:
# set(boost_address_model address-model=64)

set(boost_targets --with-filesystem --with-system --with-thread --with-date_time --with-regex --with-test --with-chrono --with-program_options)
ExternalProject_Add(boost
	URL http://downloads.sourceforge.net/project/boost/boost/1.55.0/boost_1_55_0.tar.gz
	BINARY_DIR boost-prefix/src/boost
	CONFIGURE_COMMAND ./bootstrap.bat
	BUILD_COMMAND ./b2.exe -j4 --build-type=complete link=static runtime-link=shared variant=debug,release threading=multi ${boost_addressModel} ${boost_targets} install --prefix=${ETH_DEPENDENCY_INSTALL_DIR}
	INSTALL_COMMAND cmake -E rename ${ETH_DEPENDENCY_INSTALL_DIR}/include/boost-1_55/boost ${ETH_DEPENDENCY_INSTALL_DIR}/include/boost
	)
else()

endif()

