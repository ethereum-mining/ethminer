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
/** @file ProofOfWork.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <boost/detail/endian.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <array>
#include <random>
#include <thread>
#include <libdevcore/Guards.h>
#include <libdevcore/Log.h>
#include <libdevcrypto/CryptoPP.h>
#include <libdevcrypto/FileSystem.h>
#include <libdevcore/Common.h>
#include "BlockInfo.h"
#include "Ethasher.h"
#include "ProofOfWork.h"
using namespace std;
using namespace std::chrono;

namespace dev
{
namespace eth
{

bool Ethash::verify(BlockInfo const& _header)
{
	return Ethasher::verify(_header);
}

std::pair<MineInfo, Ethash::Proof> Ethash::mine(BlockInfo const& _header, unsigned _msTimeout, bool _continue, bool _turbo)
{
	Ethasher::Miner m(_header);

	std::pair<MineInfo, Proof> ret;
	static std::mt19937_64 s_eng((time(0) + *reinterpret_cast<unsigned*>(m_last.data())));
	uint64_t tryNonce = (uint64_t)(u64)(m_last = Nonce::random(s_eng));

	bigint boundary = (bigint(1) << 256) / _header.difficulty;
	ret.first.requirement = log2((double)boundary);

	// 2^ 0      32      64      128      256
	//   [--------*-------------------------]
	//
	// evaluate until we run out of time
	auto startTime = std::chrono::steady_clock::now();
	if (!_turbo)
		std::this_thread::sleep_for(std::chrono::milliseconds(_msTimeout * 90 / 100));
	double best = 1e99;	// high enough to be effectively infinity :)
	Proof result;
	unsigned hashCount = 0;
	for (; (std::chrono::steady_clock::now() - startTime) < std::chrono::milliseconds(_msTimeout) && _continue; tryNonce++, hashCount++)
	{
		u256 val(m.mine(tryNonce));
		best = std::min<double>(best, log2((double)val));
		if (val <= boundary)
		{
			ret.first.completed = true;
			result.mixHash = m.lastMixHash();
			result.nonce = u64(tryNonce);
			break;
		}
	}
	ret.first.hashes = hashCount;
	ret.first.best = best;
	ret.second = result;

	if (ret.first.completed)
	{
		BlockInfo test = _header;
		assignResult(result, test);
		assert(verify(test));
	}

	return ret;
}

}
}
