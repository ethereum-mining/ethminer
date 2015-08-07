cmake_policy(SET CMP0015 NEW)

file(GLOB HEADERS "*.h")
include_directories(BEFORE ${JSONCPP_INCLUDE_DIRS})
include_directories(BEFORE ..)
include_directories(BEFORE ../..)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CRYPTOPP_INCLUDE_DIRS})
include_directories(${JSON_RPC_CPP_INCLUDE_DIRS})

add_executable(createRandomTest "./createRandomTest.cpp" "../TestHelper.cpp" "../Stats.cpp" "fuzzHelper.cpp" "../libdevcore/rlp.cpp" "../libethereum/transactionTests.cpp" "../libethereum/state.cpp" "../libevm/vm.cpp" "../libethereum/blockchain.cpp")

list(APPEND SRCS "./fuzzHelper.cpp")
add_sources(${SRCS})

target_link_libraries(createRandomTest ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES})
target_link_libraries(createRandomTest ethereum)
target_link_libraries(createRandomTest ethcore)
target_link_libraries(createRandomTest testutils)
