# this macro requires the following variables to be specified:
#
# ETH_DEPENDENCY_SERVER - server from which dependencies should be downloaded
# ETH_DEPENDENCY_INSTALL_DIR - install location for all dependencies
#
# usage:
#
# eth_download("json-rpc-cpp")
# eth_download("json-rpc-cpp" VERSION "0.3.2")
#
# parsing arguments
# http://www.cmake.org/cmake/help/v3.0/module/CMakeParseArguments.html
#
# for macos you may need to specify OSX_SCRIPT with install_name_tool to fix dylib
# http://stackoverflow.com/questions/2985315/using-install-name-tool-whats-going-wrong
#
# TODO: 
# check if install_command is handling symlinks correctly on linux and windows

macro(eth_download eth_package_name)

	set (extra_macro_args ${ARGN})
	set (options)
	set (one_value_args VERSION OSX_SCRIPT UNIX_SCRIPT WIN_SCRIPT)
	set (multi_value_args)
	cmake_parse_arguments (ETH_DOWNLOAD "${options}" "${one_value_args}" "${multi_value_args}" ${extra_macro_args})

	if (ETH_DOWNLOAD_VERSION)
		set(eth_tar_name "${eth_package_name}-${ETH_DOWNLOAD_VERSION}.tar.gz")
	else()
		set(eth_tar_name "${eth_package_name}.tar.gz")
	endif()

	message(STATUS "download path for ${eth_package_name} is :  ${ETH_DEPENDENCY_SERVER}/${eth_tar_name}")

	# we need that to copy symlinks
	# see http://superuser.com/questions/138587/how-to-copy-symbolic-links
	if (APPLE)
		set (eth_package_copy cp -a . ${ETH_DEPENDENCY_INSTALL_DIR})
		set (eth_package_install ${ETH_DOWNLOAD_OSX_SCRIPT})
	elseif (UNIX)
		set (eth_package_copy cp -a . ${ETH_DEPENDENCY_INSTALL_DIR})
		set (eth_package_install ${ETH_DOWNLOAD_UNIX_SCRIPT})
	else ()
		set (eth_package_copy cmake -E copy_directory . ${ETH_DEPENDENCY_INSTALL_DIR})
		set (eth_package_install ${ETH_DOWNLOAD_WIN_SCRIPT})
	endif()

	if (eth_package_install)
		message(STATUS "install script is at: ${eth_package_install}")
	endif()

	ExternalProject_Add(${eth_package_name}
		URL ${ETH_DEPENDENCY_SERVER}/${eth_tar_name}
		BINARY_DIR ${eth_package_name}-prefix/src/${eth_package_name}
		CONFIGURE_COMMAND ""
		BUILD_COMMAND ${eth_package_copy}
		INSTALL_COMMAND ${eth_package_install}
	)
endmacro()

