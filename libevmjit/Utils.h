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

namespace dev
{
namespace evmjit
{

struct Voider
{
	void operator=(std::ostream const&) {}
};

}
}


#define DLOG(CHANNEL) true ? (void)0 : ::dev::evmjit::Voider{} = std::cerr

#endif
