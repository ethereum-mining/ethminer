# all dependencies that are not directly included in the cpp-ethereum distribution are defined here
# for this to work, download the dependency via the cmake script in extdep or install them manually!

set(ETH_DEPENDENCY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extdep/install")

set (CRYPTOPP_ROOT_DIR ${ETH_DEPENDENCY_INSTALL_DIR}) 	
find_package (CryptoPP 5.6.2 REQUIRED)
message("-- CryptoPP header: ${CRYPTOPP_INCLUDE_DIRS}")
message("-- CryptoPP libs  : ${CRYPTOPP_LIBRARIES}")


