#pragma once

#include "RuntimeData.h"

namespace dev
{
namespace eth
{
namespace jit
{
using MemoryImpl = bytes;

class Runtime
{
public:
	void init(RuntimeData* _data, Env* _env);
	EXPORT ~Runtime();

	bytes_ref getReturnData() const;

private:
	RuntimeData* m_data = nullptr;	///< Pointer to data. Expected by compiled contract.
	Env* m_env = nullptr;			///< Pointer to environment proxy. Expected by compiled contract.
	byte* m_memoryData = nullptr;
	i256 m_memorySize;
	byte* m_memData;
	uint64_t m_memSize;
	uint64_t m_memCap;
};

}
}
}
