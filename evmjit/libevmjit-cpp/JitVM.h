#pragma once

#include <libevm/VMFace.h>
#include <evmjit/libevmjit/ExecutionContext.h>
#include <evmjit/libevmjit/ExecutionEngine.h>

namespace dev
{
namespace eth
{

class JitVM: public VMFace
{
	virtual bytesConstRef go(ExtVMFace& _ext, OnOpFunc const& _onOp = {}, uint64_t _steps = (uint64_t)-1) override final;

private:
	friend class VMFactory;
	explicit JitVM(u256 _gas = 0) : VMFace(_gas) {}

	jit::RuntimeData m_data;
	evmjit::ExecutionContext m_context;
	std::unique_ptr<VMFace> m_fallbackVM; ///< VM used in case of input data rejected by JIT
};


}
}
