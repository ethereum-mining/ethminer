#pragma once

#include <libevm/VMFace.h>
#include <evmjit/libevmjit/ExecutionEngine.h>

namespace dev
{
namespace eth
{

class JitVM: public VMFace
{
	virtual bytesConstRef go(ExtVMFace& _ext, OnOpFunc const& _onOp = {}, uint64_t _steps = (uint64_t)-1) override final;

	virtual u256 gas() const noexcept { return m_gas; }
	virtual void reset(u256 const& _gas = 0) noexcept { m_gas = _gas; }

private:
	friend class VMFactory;
	explicit JitVM(u256 _gas = 0): m_gas(_gas) {}

	u256 m_gas;
	jit::RuntimeData m_data;
	jit::ExecutionEngine m_engine;
	std::unique_ptr<VMFace> m_fallbackVM; ///< VM used in case of input data rejected by JIT
};


}
}
