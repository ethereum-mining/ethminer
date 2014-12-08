cmake_policy(SET CMP0015 OLD)
set(CMAKE_AUTOMOC OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATICLIB")

aux_source_directory(. SRC_LIST)

set(EXECUTABLE ethereum)

file(GLOB HEADERS "*.h")

if (ETH_STATIC)
	add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
else()
	add_library(${EXECUTABLE} SHARED ${SRC_LIST} ${HEADERS})
endif()

include_directories(${LEVELDB_INCLUDE_DIR})
include_directories(..)

target_link_libraries(${EXECUTABLE} ${LEVELDB_LIBRARY})

if (MINIUPNPC_LS)
target_link_libraries(${EXECUTABLE} ${MINIUPNPC_LS})
endif()

target_link_libraries(${EXECUTABLE} evm)
target_link_libraries(${EXECUTABLE} lll)
target_link_libraries(${EXECUTABLE} whisper)
target_link_libraries(${EXECUTABLE} p2p)
target_link_libraries(${EXECUTABLE} devcrypto)
target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} secp256k1)

install( TARGETS ${EXECUTABLE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

