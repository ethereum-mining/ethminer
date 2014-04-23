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
/** @file Dagger.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Dagger algorithm. Or not.
 */

#pragma once

#include "CommonEth.h"

#define FAKE_DAGGER 1

namespace eth
{

inline uint toLog2(u256 _d)
{
	return (uint)log2((double)_d);
}

struct MineInfo
{
	uint requirement;
	uint best;
	bool completed;
};

#if FAKE_DAGGER

class Dagger
{
public:
	static h256 eval(h256 const& _root, h256 const& _nonce) { h256 b[2] = { _root, _nonce }; return sha3(bytesConstRef((byte const*)&b[0], 64)); }
	static bool verify(h256 const& _root, h256 const& _nonce, u256 const& _difficulty) { return (bigint)(u256)eval(_root, _nonce) <= (bigint(1) << 256) / _difficulty; }

	MineInfo mine(h256& o_solution, h256 const& _root, u256 const& _difficulty, uint _msTimeout = 100, bool const& _continue = bool(true));
};

#else

/// Functions are not re-entrant. If you want to multi-thread, then use different classes for each thread.
class Dagger
{
public:
	Dagger();
	~Dagger();
	
	static u256 bound(u256 const& _difficulty);
	static h256 eval(h256 const& _root, u256 const& _nonce);
	static bool verify(h256 const& _root, u256 const& _nonce, u256 const& _difficulty);

	bool mine(u256& o_solution, h256 const& _root, u256 const& _difficulty, uint _msTimeout = 100, bool const& _continue = bool(true));

private:

	static h256 node(h256 const& _root, h256 const& _xn, uint_fast32_t _L, uint_fast32_t _i);

	h256 m_root;
	u256 m_nonce;
};

#endif

}


