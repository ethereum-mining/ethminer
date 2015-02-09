#pragma once
#include "RuntimeData.h"

namespace dev
{
namespace eth
{
namespace jit
{

using StackImpl = std::vector<i256>;
using MemoryImpl = bytes;

class Runtime
{
public:
	Runtime(RuntimeData* _data, Env* _env);

	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;

	StackImpl& getStack() { return m_stack; }
	MemoryImpl& getMemory() { return m_memory; }
	Env* getEnvPtr() { return &m_env; }

	bytes_ref getReturnData() const;

private:
	RuntimeData& m_data;			///< Pointer to data. Expected by compiled contract.
	Env& m_env;						///< Pointer to environment proxy. Expected by compiled contract.
	void* m_currJmpBuf = nullptr;	///< Pointer to jump buffer. Expected by compiled contract.
	byte* m_memoryData = nullptr;
	i256 m_memorySize;
	StackImpl m_stack;
	MemoryImpl m_memory;
};

}
}
}
