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
	Runtime(RuntimeData* _data, Env* _env);

	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;

	MemoryImpl& getMemory() { return m_memory; }

	bytes_ref getReturnData() const;

private:
	RuntimeData& m_data;			///< Pointer to data. Expected by compiled contract.
	Env& m_env;						///< Pointer to environment proxy. Expected by compiled contract.
	byte* m_memoryData = nullptr;
	i256 m_memorySize;
	byte* m_memData = nullptr;
	uint64_t m_memSize = 0;
	uint64_t m_memCap = 0;
	MemoryImpl m_memory;
};

}
}
}
