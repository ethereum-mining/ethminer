#pragma once

#include <libevm/VMFace.h>
#include <evmjit/JIT.h>

namespace dev
{
namespace eth
{

class JitVM: public VMFace
{
public:
	virtual bytesConstRef execImpl(u256& io_gas, ExtVMFace& _ext, OnOpFunc const& _onOp) override final;

private:
	evmjit::RuntimeData m_data;
	evmjit::ExecutionContext m_context;
	std::unique_ptr<VMFace> m_fallbackVM; ///< VM used in case of input data rejected by JIT
};


}
}
