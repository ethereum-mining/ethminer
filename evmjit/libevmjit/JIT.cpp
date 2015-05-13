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
	std::unordered_map<h256, void*> codeMap;

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

void* JIT::getCode(h256 _codeHash)
{
	auto& codeMap = JITImpl::instance().codeMap;
	auto it = codeMap.find(_codeHash);
	if (it != codeMap.end())
		return it->second;
	return nullptr;
}

void JIT::mapCode(h256 _codeHash, void* _funcAddr)
{
	JITImpl::instance().codeMap.insert(std::make_pair(_codeHash, _funcAddr));
}

}
}
