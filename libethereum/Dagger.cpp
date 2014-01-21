#include <boost/detail/endian.hpp>
#include <chrono>
#include <array>
#if WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#else
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include <sha3.h>
#if WIN32
#pragma warning(pop)
#endif
#include <random>
#include "Common.h"
#include "Dagger.h"
using namespace std;
using namespace std::chrono;

namespace eth
{

#if FAKE_DAGGER

bool Dagger::mine(u256& o_solution, h256 const& _root, u256 const& _difficulty, uint _msTimeout, bool const& _continue)
{
	static std::mt19937_64 s_eng((time(0)));
	o_solution = std::uniform_int_distribution<uint>(0, ~(uint)0)(s_eng);
	// evaluate until we run out of time
	for (auto startTime = steady_clock::now(); (steady_clock::now() - startTime) < milliseconds(_msTimeout) && _continue; o_solution += 1)
		if (verify(_root, o_solution, _difficulty))
			return true;
	return false;
}

#else

Dagger::Dagger()
{
}

Dagger::~Dagger()
{
}

u256 Dagger::bound(u256 const& _difficulty)
{
	return (u256)((bigint(1) << 256) / _difficulty);
}

bool Dagger::verify(h256 const& _root, u256 const& _nonce, u256 const& _difficulty)
{
	return eval(_root, _nonce) < bound(_difficulty);
}

bool Dagger::mine(u256& o_solution, h256 const& _root, u256 const& _difficulty, uint _msTimeout, bool const& _continue)
{
	// restart search if root has changed
	if (m_root != _root)
	{
		m_root = _root;
		m_nonce = 0;
	}

	// compute bound
	u256 const b = bound(_difficulty);

	// evaluate until we run out of time
	for (auto startTime = steady_clock::now(); (steady_clock::now() - startTime) < milliseconds(_msTimeout) && _continue; m_nonce += 1)
	{
		if (eval(_root, m_nonce) < b)
		{
			o_solution = m_nonce;
			return true;
		}
	}
	return false;
}

template <class _T>
inline void update(_T& _sha, u256 const& _value)
{
	int i = 0;
	for (u256 v = _value; v; ++i, v >>= 8) {}
	byte buf[32];
	bytesRef bufRef(buf, i);
	toBigEndian(_value, bufRef);
	_sha.Update(buf, i);
}

template <class _T>
inline void update(_T& _sha, h256 const& _value)
{
	int i = 0;
	byte const* data = _value.data();
	for (; i != 32 && data[i] == 0; ++i);
	_sha.Update(data + i, 32 - i);
}

template <class _T>
inline h256 get(_T& _sha)
{
	h256 ret;
	_sha.TruncatedFinal(&ret[0], 32);
	return ret;
}

h256 Dagger::node(h256 const& _root, h256 const& _xn, uint_fast32_t _L, uint_fast32_t _i)
{
	if (_L == _i)
		return _root;
	u256 m = (_L == 9) ? 16 : 3;
	CryptoPP::SHA3_256 bsha;
	for (uint_fast32_t k = 0; k < m; ++k)
	{
		CryptoPP::SHA3_256 sha;
		update(sha, _root);
		update(sha, _xn);
		update(sha, (u256)_L);
		update(sha, (u256)_i);
		update(sha, (u256)k);
		uint_fast32_t pk = (uint_fast32_t)(u256)get(sha) & ((1 << ((_L - 1) * 3)) - 1);
		auto u = node(_root, _xn, _L - 1, pk);
		update(bsha, u);
	}
	return get(bsha);
}

h256 Dagger::eval(h256 const& _root, u256 const& _nonce)
{
	h256 extranonce = _nonce >> 26;				// with xn = floor(n / 2^26) -> assuming this is with xn = floor(N / 2^26)
	CryptoPP::SHA3_256 bsha;
	for (uint_fast32_t k = 0; k < 4; ++k)
	{
		//sha256(D || xn || i || k)		-> sha256(D || xn || k)	- there's no 'i' here!
		CryptoPP::SHA3_256 sha;
		update(sha, _root);
		update(sha, extranonce);
		update(sha, _nonce);
		update(sha, (u256)k);
		uint_fast32_t pk = (uint_fast32_t)(u256)get(sha) & 0x1ffffff;	// mod 8^8 * 2  [ == mod 2^25 ?! ] [ == & ((1 << 25) - 1) ] [ == & 0x1ffffff ]
		auto u = node(_root, extranonce, 9, pk);
		update(bsha, u);
	}
	return get(bsha);
}

#endif
}
