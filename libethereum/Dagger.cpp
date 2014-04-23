/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Dagger.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <boost/detail/endian.hpp>
#include <chrono>
#include <array>
#include <random>
#include <libethcore/CryptoHeaders.h>
#include <libethcore/Common.h>
#include "Dagger.h"
using namespace std;
using namespace std::chrono;

namespace eth
{

#if FAKE_DAGGER

MineInfo Dagger::mine(h256& o_solution, h256 const& _root, u256 const& _difficulty, uint _msTimeout, bool const& _continue)
{
	MineInfo ret{0, 0, false};
	static std::mt19937_64 s_eng((time(0)));
	u256 s = std::uniform_int_distribution<uint>(0, ~(uint)0)(s_eng);

	bigint d = (bigint(1) << 256) / _difficulty;
	ret.requirement = toLog2((u256)d);

	// 2^ 0      32      64      128      256
	//   [--------*-------------------------]
	//
	// evaluate until we run out of time
	for (auto startTime = steady_clock::now(); (steady_clock::now() - startTime) < milliseconds(_msTimeout) && _continue; s++)
	{
		o_solution = (h256)s;
		auto e = (bigint)(u256)eval(_root, o_solution);
		ret.best = max(ret.best, toLog2((u256)e));
		if (e <= d)
		{
			ret.completed = true;
			break;
		}
	}

	if (ret.completed)
		assert(verify(_root, o_solution, _difficulty));

	return ret;
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
