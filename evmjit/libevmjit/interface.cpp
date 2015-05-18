#include "evmjit/JIT.h"
#include "ExecutionContext.h"

extern "C"
{
using namespace dev::evmjit;

EXPORT void* evmjit_create(RuntimeData* _data, Env* _env) noexcept
{
	if (!_data)
		return nullptr;

	// TODO: Make sure ExecutionEngine constructor does not throw + make JIT/ExecutionEngine interface all nothrow
	return new(std::nothrow) ExecutionContext{*_data, _env};
}

EXPORT void evmjit_destroy(ExecutionContext* _context) noexcept
{
	delete _context;
}

EXPORT int evmjit_run(ExecutionContext* _context) noexcept
{
	try
	{
		auto returnCode = JIT::exec(*_context);
		return static_cast<int>(returnCode);
	}
	catch(...)
	{
		return static_cast<int>(ReturnCode::UnexpectedException);
	}
}

}
