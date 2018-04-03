add_library(
    ethash-legacy
    internal.c
    ethash.h
    endian.h
    compiler.h
    fnv.h
    data_sizes.h
    sha3.c
    sha3.h
)
target_compile_options(ethash-legacy PRIVATE -std=gnu99)
