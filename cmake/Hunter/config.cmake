hunter_config(CURL VERSION ${HUNTER_CURL_VERSION} CMAKE_ARGS HTTP_ONLY=ON CMAKE_USE_OPENSSL=OFF CMAKE_USE_LIBSSH2=OFF CURL_CA_PATH=none)
hunter_config(
    Boost
    URL "https://boostorg.jfrog.io/ui/api/v1/download?repoKey=main&path=release%252F1.77.0%252Fsource%252Fboost_1_77_0.tar.gz"
    SHA1 "7f906921bffea1a84b45e39c092c317dcc5794f4"
)

hunter_config(
    OpenSSL
    URL "https://www.openssl.org/source/openssl-1.1.1j.tar.gz"
    SHA1 "04c340b086828eecff9df06dceff196790bb9268"
    CMAKE_ARGS configure_architectures=arm64
)
