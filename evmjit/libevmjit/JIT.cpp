#include "evmjit/JIT.h"

#include <unordered_map>

namespace dev
{
namespace evmjit
{
namespace
{

class JITImpl: JIT
{
public:
	std::unordered_map<h256, uint64_t> codeMap;

	static JITImpl& instance()
	{
		static JITImpl s_instance;
		return s_instance;
	}
};

} // anonymous namespace

bool JIT::isCodeReady(h256 _codeHash)
{
	return JITImpl::instance().codeMap.count(_codeHash) != 0;
}

uint64_t JIT::getCode(h256 _codeHash)
{
	auto& codeMap = JITImpl::instance().codeMap;
	auto it = codeMap.find(_codeHash);
	if (it != codeMap.end())
		return it->second;
	return 0;
}

void JIT::mapCode(h256 _codeHash, uint64_t _funcAddr)
{
	JITImpl::instance().codeMap.insert(std::make_pair(_codeHash, _funcAddr));
}

}
}
