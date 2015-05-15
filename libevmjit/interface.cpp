#include "ExecutionEngine.h"

extern "C"
{

using namespace dev::eth::jit;

EXPORT void* evmjit_create() noexcept
{
	// TODO: Make sure ExecutionEngine constructor does not throw + make JIT/ExecutionEngine interface all nothrow
	return new(std::nothrow) ExecutionContext;
}

EXPORT void evmjit_destroy(ExecutionContext* _context) noexcept
{
	delete _context;
}

EXPORT int evmjit_run(ExecutionContext* _context, RuntimeData* _data, Env* _env) noexcept
{
	if (!_context || !_data)
		return static_cast<int>(ReturnCode::UnexpectedException);

	try
	{
		auto returnCode = ExecutionEngine::run(*_context, _data, _env);
		return static_cast<int>(returnCode);
	}
	catch(...)
	{
		return static_cast<int>(ReturnCode::UnexpectedException);
	}
}

}
