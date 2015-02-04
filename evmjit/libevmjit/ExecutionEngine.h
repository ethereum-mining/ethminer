#pragma once

#include <memory>
#include "RuntimeData.h"
#include "ExecStats.h"

namespace dev
{
namespace eth
{
namespace jit
{

class ExecutionEngine
{
public:
	ExecutionEngine() = default;
	ExecutionEngine(ExecutionEngine const&) = delete;
	void operator=(ExecutionEngine) = delete;

	EXPORT ReturnCode run(RuntimeData* _data, Env* _env);

	void collectStats();

	std::unique_ptr<ExecStats> getStats();

	/// Reference to returned data (RETURN opcode used)
	bytes_ref returnData;

private:
	/// After execution, if RETURN used, memory is moved there
	/// to allow client copy the returned data
	bytes m_memory;

	std::unique_ptr<ExecStats> m_stats;
};

}
}
}
