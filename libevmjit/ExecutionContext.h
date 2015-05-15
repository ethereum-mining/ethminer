#pragma once

#include "RuntimeData.h"

namespace dev
{
namespace evmjit
{
	using namespace eth::jit; // FIXME

class ExecutionContext
{
public:
	ExecutionContext() = default;
	ExecutionContext(RuntimeData& _data, Env* _env) { init(_data, _env); }
	ExecutionContext(ExecutionContext const&) = delete;
	ExecutionContext& operator=(ExecutionContext const&) = delete;
	EXPORT ~ExecutionContext();

	void init(RuntimeData& _data, Env* _env) { m_data = &_data; m_env = _env; }

	byte const* code() const { return m_data->code; }
	uint64_t codeSize() const { return m_data->codeSize; }
	h256 const& codeHash() const { return m_data->codeHash; }

	bytes_ref getReturnData() const;

private:
	RuntimeData* m_data = nullptr;	///< Pointer to data. Expected by compiled contract.
	Env* m_env = nullptr;			///< Pointer to environment proxy. Expected by compiled contract.
	byte* m_memData = nullptr;
	uint64_t m_memSize = 0;
	uint64_t m_memCap = 0;

public:
	/// Reference to returned data (RETURN opcode used)
	bytes_ref returnData;
};

}
}
