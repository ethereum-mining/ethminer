#include "ExecutionEngine.h"

extern "C"
{

using namespace dev::eth::jit;

EXPORT void* evmjit_create() noexcept
{
	// TODO: Make sure ExecutionEngine constructor does not throw
	return new(std::nothrow) ExecutionEngine;
}

EXPORT void evmjit_destroy(ExecutionEngine* _engine) noexcept
{
	delete _engine;
}

EXPORT int evmjit_run(ExecutionEngine* _engine, RuntimeData* _data, Env* _env) noexcept
{
	if (!_engine || !_data)
		return static_cast<int>(ReturnCode::UnexpectedException);

	try
	{
		auto returnCode = _engine->run(_data, _env);
		return static_cast<int>(returnCode);
	}
	catch(...)
	{
		return static_cast<int>(ReturnCode::UnexpectedException);
	}
}

}
