#pragma once

#include <vector>
#include <tuple>
#include <cstdint>

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
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
using bytes = std::vector<byte>;
using bytes_ref = std::tuple<byte const*, size_t>;

struct NoteChannel {};	// FIXME: Use some log library?

enum class ReturnCode
{
	Stop = 0,
	Return = 1,
	Suicide = 2,

	OutOfGas = -1,
	BadJumpDestination = -2,
	StackTooSmall = -3,
	BadInstruction = -4,

	LLVMConfigError = -5,
	LLVMCompileError = -6,
	LLVMLinkError = -7,

	UnexpectedException = -8,

	LinkerWorkaround = -299,
};

/// Representation of 256-bit value binary compatible with LLVM i256
struct i256
{
	uint64_t a = 0;
	uint64_t b = 0;
	uint64_t c = 0;
	uint64_t d = 0;
};
static_assert(sizeof(i256) == 32, "Wrong i265 size");

#define UNTESTED assert(false)

}
}
}
