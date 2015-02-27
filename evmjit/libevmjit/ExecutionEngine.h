#pragma once

#include <memory>

#include "Runtime.h"

namespace dev
{
namespace eth
{
namespace jit
{

enum class ExecState
{
	Started,
	CacheLoad,
	CacheWrite,
	Compilation,
	Optimization,
	CodeGen,
	Execution,
	Return,
	Finished
};

class ExecutionEngineListener
{
public:
	ExecutionEngineListener() = default;
	ExecutionEngineListener(ExecutionEngineListener const&) = delete;
	ExecutionEngineListener& operator=(ExecutionEngineListener) = delete;
	virtual ~ExecutionEngineListener() {}

	virtual void executionStarted() {}
	virtual void executionEnded() {}

	virtual void stateChanged(ExecState) {}
};

class ExecutionEngine
{
public:
	ExecutionEngine() = default;
	ExecutionEngine(ExecutionEngine const&) = delete;
	ExecutionEngine& operator=(ExecutionEngine) = delete;

	EXPORT ReturnCode run(RuntimeData* _data, Env* _env);

	/// Reference to returned data (RETURN opcode used)
	bytes_ref returnData;

private:
	Runtime m_runtime;
};

}
}
}
