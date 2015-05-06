cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)
include_directories(${LEVELDB_INCLUDE_DIRS})

set(EXECUTABLE rlp)

add_executable(${EXECUTABLE} ${SRC_LIST})

target_link_libraries(${EXECUTABLE} devcrypto)

eth_install_executable(${EXECUTABLE})
