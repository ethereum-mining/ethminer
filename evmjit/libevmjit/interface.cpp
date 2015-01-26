#include "ExecutionEngine.h"

extern "C"
{

using namespace dev::eth::jit;

void* evmjit_create() noexcept
{
	return new(std::nothrow) ExecutionEngine;
}

void evmjit_destroy(ExecutionEngine* _engine) noexcept
{
	delete _engine;
}

int evmjit_run(ExecutionEngine* _engine, RuntimeData* _data, Env* _env) noexcept
{
	try
	{
		auto codePtr = _data->code;
		auto codeSize = _data->codeSize;
		bytes bytecode;
		bytecode.insert(bytecode.end(), codePtr, codePtr + codeSize);

		auto returnCode = _engine->run(bytecode, _data, _env);
		return static_cast<int>(returnCode);
	}
	catch(...)
	{
		return static_cast<int>(ReturnCode::UnexpectedException);
	}
}

}
