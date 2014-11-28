#pragma once

#include <libevm/VMFace.h>

namespace dev
{
namespace eth
{

/**
 */

class VMFactory
{
public:
	enum Kind: bool {
		Interpreter,
#if ETH_EVMJIT
		JIT
#endif
	};

	static std::unique_ptr<VMFace> create(Kind, u256 _gas = 0);
};


}
}
