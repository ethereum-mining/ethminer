#pragma once

#include <iostream>

#include "Common.h"

namespace dev
{
namespace eth
{
namespace jit
{

struct JIT: public NoteChannel  { static const char* name() { return "JIT"; } };

//#define clog(CHANNEL) std::cerr
#define clog(CHANNEL) std::ostream(nullptr)

// The same as assert, but expression is always evaluated and result returned
#define CHECK(expr) (assert(expr), expr)

}
}
}
