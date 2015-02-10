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
/** @file ProofOfWork.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * ProofOfWork algorithm. Or not.
 */

#pragma once

#include <chrono>
#include <thread>
#include <cstdint>
#include <libdevcrypto/SHA3.h>
#include "CommonEth.h"

#define FAKE_DAGGER 1

namespace dev
{
namespace eth
{

struct MineInfo
{
	void combine(MineInfo const& _m) { requirement = std::max(requirement, _m.requirement); best = std::min(best, _m.best); hashes += _m.hashes; completed = completed || _m.completed; }
	double requirement = 0;
	double best = 1e99;
	unsigned hashes = 0;
	bool completed = false;
};

template <class Evaluator>
class ProofOfWorkEngine: public Evaluator
{
public:
	static bool verify(h256 const& _root, h256 const& _nonce, u256 const& _difficulty) { return (bigint)(u256)Evaluator::eval(_root, _nonce) <= (bigint(1) << 256) / _difficulty; }

	inline std::pair<MineInfo, h256> mine(h256 const& _root, u256 const& _difficulty, unsigned _msTimeout = 100, bool _continue = true, bool _turbo = false);

protected:
	h256 m_last;
};

class SHA3Evaluator
{
public:
	static h256 eval(h256 const& _root, h256 const& _nonce) { h256 b[2] = { _root, _nonce }; return sha3(bytesConstRef((byte const*)&b[0], 64)); }
};

// TODO: class ARPoWEvaluator

class DaggerEvaluator
{
public:
	static h256 eval(h256 const& _root, h256 const& _nonce);

private:
	static h256 node(h256 const& _root, h256 const& _xn, uint_fast32_t _L, uint_fast32_t _i);
};

using SHA3ProofOfWork = ProofOfWorkEngine<SHA3Evaluator>;

using ProofOfWork = SHA3ProofOfWork;

template <class Evaluator>
std::pair<MineInfo, h256> ProofOfWorkEngine<Evaluator>::mine(h256 const& _root, u256 const& _difficulty, unsigned _msTimeout, bool _continue, bool _turbo)
{
	std::pair<MineInfo, h256> ret;
	static std::mt19937_64 s_eng((time(0) + (unsigned)m_last));
	u256 s = (m_last = h256::random(s_eng));

	bigint d = (bigint(1) << 256) / _difficulty;
	ret.first.requirement = log2((double)d);

	// 2^ 0      32      64      128      256
	//   [--------*-------------------------]
	//
	// evaluate until we run out of time
	auto startTime = std::chrono::steady_clock::now();
	if (!_turbo)
		std::this_thread::sleep_for(std::chrono::milliseconds(_msTimeout * 90 / 100));
	double best = 1e99;	// high enough to be effectively infinity :)
	h256 solution;
	unsigned h = 0;
	for (; (std::chrono::steady_clock::now() - startTime) < std::chrono::milliseconds(_msTimeout) && _continue; s++, h++)
	{
		solution = (h256)s;
		auto e = (bigint)(u256)Evaluator::eval(_root, solution);
		best = std::min<double>(best, log2((double)e));
		if (e <= d)
		{
			ret.first.completed = true;
			break;
		}
	}
	ret.first.hashes = h;
	ret.first.best = best;
	ret.second = solution;

	if (ret.first.completed)
		assert(verify(_root, solution, _difficulty));

	return ret;
}

}
}
