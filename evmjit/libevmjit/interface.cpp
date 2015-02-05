#include "ExecutionEngine.h"

extern "C"
{

using namespace dev::eth::jit;

#ifdef _MSC_VER
#define _ALLOW_KEYWORD_MACROS
#define noexcept throw()
#endif

EXPORT void* evmjit_create() noexcept
{
	return new(std::nothrow) ExecutionEngine;
}

EXPORT void evmjit_destroy(ExecutionEngine* _engine) noexcept
{
	delete _engine;
}

EXPORT int evmjit_run(ExecutionEngine* _engine, RuntimeData* _data, Env* _env) noexcept
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
