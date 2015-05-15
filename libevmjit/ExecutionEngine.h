#pragma once

#include <memory>
#include "Common.h"


namespace dev
{
namespace evmjit
{
	class ExecutionContext;
}
namespace eth
{
namespace jit
{
	using namespace evmjit; // FIXME

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
	ExecutionEngine(ExecutionEngine const&) = delete;
	ExecutionEngine& operator=(ExecutionEngine const&) = delete;

	EXPORT static ReturnCode run(ExecutionContext& _context);

private:
	ExecutionEngine();
	static ExecutionEngine& get();
};

}
}
}
