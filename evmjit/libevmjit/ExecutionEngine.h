#pragma once

#include "Common.h"

namespace dev
{
namespace evmjit
{
class ExecutionContext;

using namespace eth::jit;

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
	EXPORT static ReturnCode run(ExecutionContext& _context);
};

}
}
