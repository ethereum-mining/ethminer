#include "Runtime.h"

#include <cassert>

namespace dev
{
namespace eth
{
namespace jit
{

void Runtime::init(RuntimeData* _data, Env* _env)
{
	m_data = _data;
	m_env = _env;
}

extern "C" void ext_free(void* _data) noexcept;

Runtime::~Runtime()
{
	if (m_memData)
		ext_free(m_memData); // Use helper free to check memory leaks
}

bytes_ref Runtime::getReturnData() const
{
	auto data = m_data->callData;
	auto size = static_cast<size_t>(m_data->callDataSize);

	if (data < m_memData || data >= m_memData + m_memSize || size == 0)
	{
		assert(size == 0); // data can be an invalid pointer only if size is 0
		m_data->callData = nullptr;
		return {};
	}

	return bytes_ref{data, size};
}

}
}
}
