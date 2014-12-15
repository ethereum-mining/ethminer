if (APPLE)

elseif (WIN32)
	ExternalProject_Add(icu
		GIT_REPOSITORY https://github.com/debris/icu-win32.git
		GIT_TAG master
		BINARY_DIR icu-prefix/src/icu
		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND cmake -E copy_directory . ${ETH_DEPENDENCY_INSTALL_DIR}
	)

else()

endif()


