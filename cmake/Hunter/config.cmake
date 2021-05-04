hunter_config(
    CURL
    VERSION ${HUNTER_CURL_VERSION} 
    CMAKE_ARGS HTTP_ONLY=ON CMAKE_USE_OPENSSL=OFF CMAKE_USE_LIBSSH2=OFF CURL_CA_PATH=none
)

hunter_config(
    Boost
    VERSION 1.76.0
    SHA1 a5ab6eaf31d1ca181a17ecffef9d58d40d87c71d
    URL https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
)

hunter_config(
    ethash
    VERSION 0.6.0
    SHA1 4bfa26b389d1f89a60053de04b2a29feab20f67b
    URL https://github.com/chfast/ethash/archive/refs/tags/v0.6.0.tar.gz
)
