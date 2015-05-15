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

class ExecutionContext
{
public:
	ExecutionContext() = default;
	ExecutionContext(ExecutionContext const&) = delete;
	ExecutionContext& operator=(ExecutionContext const&) = delete;

	/// Reference to returned data (RETURN opcode used)
	bytes_ref returnData;

	Runtime m_runtime;
};

class ExecutionEngine
{
public:
	ExecutionEngine(ExecutionEngine const&) = delete;
	ExecutionEngine& operator=(ExecutionEngine const&) = delete;

	EXPORT static ReturnCode run(ExecutionContext& _context, RuntimeData* _data, Env* _env);

private:
	ExecutionEngine();
	static ExecutionEngine& get();
};

}
}
}
