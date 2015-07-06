#pragma once

#include <evmjit/JIT.h>

namespace dev
{
namespace eth
{

/// Converts EVM JIT representation of 256-bit integer to eth type dev::u256.
inline u256 jit2eth(evmjit::i256 _i)
{
	u256 u = _i.words[3];
	u <<= 64;
	u |= _i.words[2];
	u <<= 64;
	u |= _i.words[1];
	u <<= 64;
	u |= _i.words[0];
	return u;
}

/// Converts eth type dev::u256 to EVM JIT representation of 256-bit integer.
inline evmjit::i256 eth2jit(u256 _u)
{
	evmjit::i256 i;
	i.words[0] = static_cast<uint64_t>(_u);
	_u >>= 64;
	i.words[1] = static_cast<uint64_t>(_u);
	_u >>= 64;
	i.words[2] = static_cast<uint64_t>(_u);
	_u >>= 64;
	i.words[3] = static_cast<uint64_t>(_u);
	return i;
}

/// Converts eth type dev::h256 to EVM JIT representation of 256-bit hash value.
inline evmjit::h256 eth2jit(h256 _u)
{
	/// Just directly copies memory
	return *(evmjit::h256*)&_u;
}

}
}
