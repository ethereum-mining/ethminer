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
set_target_properties(ethash-legacy PROPERTIES C_STANDARD 99)
