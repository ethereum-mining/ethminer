
#include "Runtime.h"

#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IntrinsicInst.h>

namespace dev
{
namespace eth
{
namespace jit
{

Runtime::Runtime(RuntimeData* _data, Env* _env) :
	m_data(*_data),
	m_env(*_env),
	m_currJmpBuf(m_jmpBuf)
{}

bytes_ref Runtime::getReturnData() const
{
	auto data = m_data.callData;
	auto size = static_cast<size_t>(m_data.callDataSize);

	if (data < m_memory.data() || data >= m_memory.data() + m_memory.size() || size == 0)
	{
		assert(size == 0); // data can be an invalid pointer only if size is 0
		m_data.callData = nullptr;
		return {};
	}

	return bytes_ref{data, size};
}

}
}
}
