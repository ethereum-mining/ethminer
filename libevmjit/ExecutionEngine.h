#pragma once

#include "evmjit/DataTypes.h"

namespace dev
{
namespace evmjit
{
class ExecutionContext;
enum class ReturnCode;

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
