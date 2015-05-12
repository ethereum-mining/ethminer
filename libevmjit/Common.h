#pragma once

#include <tuple>
#include <cstdint>

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#define _ALLOW_KEYWORD_MACROS
#define noexcept throw()
#else
#define EXPORT
#endif

namespace dev
{
namespace eth
{
namespace jit
{

using byte = uint8_t;
using bytes_ref = std::tuple<byte const*, size_t>;
using code_iterator = byte const*;

enum class ReturnCode
{
	// Success codes
	Stop    = 0,
	Return  = 1,
	Suicide = 2,

	// Standard error codes
	OutOfGas           = -1,
	StackUnderflow     = -2,
	BadJumpDestination = -3,
	BadInstruction     = -4,
	Rejected           = -5, ///< Input data (code, gas, block info, etc.) does not meet JIT requirement and execution request has been rejected

	// Internal error codes
	LLVMConfigError    = -101,
	LLVMCompileError   = -102,
	LLVMLinkError      = -103,

	UnexpectedException = -111,

	LinkerWorkaround = -299,
};

#define UNTESTED assert(false)

}
}
}
