#define FNV_PRIME 0x01000193

#define fnv(x, y) ((x)*FNV_PRIME ^ (y))

DEV_INLINE uint4 fnv4(uint4 a, uint4 b)
{
    
    //c.x = a.x * FNV_PRIME ^ b.x;
    //c.y = a.y * FNV_PRIME ^ b.y;
    //c.z = a.z * FNV_PRIME ^ b.z;
    //c.w = a.w * FNV_PRIME ^ b.w;

    uint4 c;
    asm("{\n\t"
        "mul.lo.u32 %0, %1, 0x01000193;\n\t"
        "xor.b32 %0, %2, %0;\n\t"
        "}\n\t"
        : "=r"(c.x)
        : "r"(a.x), "r"(b.x));

    asm("{\n\t"
        "mul.lo.u32 %0, %1, 0x01000193;\n\t"
        "xor.b32 %0, %2, %0;\n\t"
        "}\n\t"
        : "=r"(c.y)
        : "r"(a.y), "r"(b.y));

    asm("{\n\t"
        "mul.lo.u32 %0, %1, 0x01000193;\n\t"
        "xor.b32 %0, %2, %0;\n\t"
        "}\n\t"
        : "=r"(c.z)
        : "r"(a.z), "r"(b.z));

    asm("{\n\t"
        "mul.lo.u32 %0, %1, 0x01000193;\n\t"
        "xor.b32 %0, %2, %0;\n\t"
        "}\n\t"
        : "=r"(c.w)
        : "r"(a.w), "r"(b.w));
    return c;
}

DEV_INLINE uint32_t fnv_reduce(uint4 v)
{
	//return fnv(fnv(fnv(v.x, v.y), v.z), v.w)
	uint32_t result;
    asm("{\n\t"
        "mul.lo.u32 %0, %1, 0x01000193;\n\t"
        "xor.b32 %0, %0, %2;\n\t"
        "mul.lo.u32 %0, %0, 0x01000193;\n\t"
        "xor.b32 %0, %0, %3;\n\t"
        "mul.lo.u32 %0, %0, 0x01000193;\n\t"
        "xor.b32 %0, %0, %4;\n\t"
        "}\n\t"
        : "=r"(result)
        : "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    return result;
}
