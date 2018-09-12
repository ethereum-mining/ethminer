
#define FNV1A_PRIME	0x010001A7

#define fnv1a(x,y) (((x) ^(y)) * FNV1A_PRIME)

__device__ uint4 fnv1a4(uint4 a, uint4 b)
{
	uint4 c;
	c.x = (a.x ^ b.x) * FNV1A_PRIME ;
	c.y = (a.y ^ b.y) * FNV1A_PRIME ;
	c.z = (a.z ^ b.z) * FNV1A_PRIME ;
	c.w = (a.w ^ b.w) * FNV1A_PRIME ;
	return c;
}

__device__ uint32_t fnv1a_reduce(uint4 v)
{
	return fnv1a(fnv1a(fnv1a(v.x, v.y), v.z), v.w);
}

