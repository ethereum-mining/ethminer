# this macro requires the following variables to be specified:
#
# ETH_DEPENDENCY_SERVER - server from which dependencies should be downloaded
# ETH_DEPENDENCY_INSTALL_DIR - install location for all dependencies
#
# usage:
#
# eth_download("json-rpc-cpp")
# eth_download("json-rpc-cpp" "0.3.2")
#
# TODO: 
# check if install_command is handling symlinks correctly on linux and windows

macro(eth_download eth_package_name)

	set (extra_macro_args ${ARGN})
	if (extra_macro_args GREATER 0)
		set(eth_tar_name "${eth_package_name}-${ARGV1}.tar.gz")
	else()
		set(eth_tar_name "${eth_package_name}.tar.gz")
	endif()

	message(STATUS "download path for ${eth_package_name} is :  ${ETH_DEPENDENCY_SERVER}/${eth_tar_name}.tar.gz")

	# we need that to copy symlinks
	# see http://superuser.com/questions/138587/how-to-copy-symbolic-links
	if (APPLE)
		set (eth_package_install cp -a . ${ETH_DEPENDENCY_INSTALL_DIR})
	elseif (UNIX)
		set (eth_package_install cp -a . ${ETH_DEPENDENCY_INSTALL_DIR})
	else ()
		set (eth_package_install cmake -E copy_directory . ${ETH_DEPENDENCY_INSTALL_DIR})
	endif()


	ExternalProject_Add(${eth_package_name}
		URL ${ETH_DEPENDENCY_SERVER}/${eth_tar_name}
		BINARY_DIR ${eth_package_name}-prefix/src/${eth_package_name}
		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND ${eth_package_install}
	)
endmacro()

