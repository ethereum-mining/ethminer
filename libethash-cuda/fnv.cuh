
#define FNV_PRIME	0x01000193

#define fnv(x,y) ((x) * FNV_PRIME ^(y))

__device__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 c;
	c.x = a.x * FNV_PRIME ^ b.x;
	c.y = a.y * FNV_PRIME ^ b.y;
	c.z = a.z * FNV_PRIME ^ b.z;
	c.w = a.w * FNV_PRIME ^ b.w;
	return c;
}

__device__ uint32_t fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

