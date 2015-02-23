#pragma once

#include <iostream>

// The same as assert, but expression is always evaluated and result returned
#define CHECK(expr) (assert(expr), expr)

#if !defined(NDEBUG) // Debug

namespace dev
{
namespace evmjit
{

std::ostream& getLogStream(char const* _channel);

}
}

#define DLOG(CHANNEL) ::dev::evmjit::getLogStream(#CHANNEL)

#else // Release
	#define DLOG(CHANNEL) true ? std::cerr : std::cerr
#endif
