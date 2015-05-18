#pragma once

#include <cstdint>
#include <functional>

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#define _ALLOW_KEYWORD_MACROS
#define noexcept throw()
#else
#define EXPORT
#endif

namespace dev
{
namespace evmjit
{

using byte = uint8_t;
using bytes_ref = std::tuple<byte const*, size_t>;
using code_iterator = byte const*;

struct h256
{
	uint64_t words[4];
};

inline bool operator==(h256 _h1, h256 _h2)
{
	return 	_h1.words[0] == _h2.words[0] &&
			_h1.words[1] == _h2.words[1] &&
			_h1.words[2] == _h2.words[2] &&
			_h1.words[3] == _h2.words[3];
}

/// Representation of 256-bit value binary compatible with LLVM i256
struct i256
{
	uint64_t a = 0;
	uint64_t b = 0;
	uint64_t c = 0;
	uint64_t d = 0;

	i256() = default;
	i256(h256 _h)
	{
		a = _h.words[0];
		b = _h.words[1];
		c = _h.words[2];
		d = _h.words[3];
	}
};

}
}

namespace std
{
template<> struct hash<dev::evmjit::h256>
{
	size_t operator()(dev::evmjit::h256 const& _h) const
	{
		/// This implementation expects the argument to be a full 256-bit Keccak hash.
		/// It does nothing more than returning a slice of the input hash.
		return static_cast<size_t>(_h.words[0]);
	};
};
}
