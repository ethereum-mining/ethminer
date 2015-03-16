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
#if ETH_ETHASHCL
#include <libethash-cl/ethash_cl_miner.h>
#endif
#include "BlockInfo.h"
#include "Ethasher.h"
#include "ProofOfWork.h"
using namespace std;
using namespace std::chrono;

namespace dev
{
namespace eth
{

bool EthashCPU::verify(BlockInfo const& _header)
{
	return Ethasher::verify(_header);
}

std::pair<MineInfo, EthashCPU::Proof> EthashCPU::mine(BlockInfo const& _header, unsigned _msTimeout, bool _continue, bool _turbo)
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

#if ETH_ETHASHCL

/*
struct ethash_cl_search_hook
{
	// reports progress, return true to abort
	virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
	virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
};

class ethash_cl_miner
{
public:
	ethash_cl_miner();

	bool init(ethash_params const& params, const uint8_t seed[32], unsigned workgroup_size = 64);

	void hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count);
	void search(uint8_t const* header, uint64_t target, search_hook& hook);
};
*/

struct EthashCLHook: public ethash_cl_search_hook
{
	virtual bool found(uint64_t const* _nonces, uint32_t _count)
	{
		Guard l(x_all);
		for (unsigned i = 0; i < _count; ++i)
			found.push_back((Nonce)(u64)_nonces[i]);
		if (abort)
		{
			aborted = true;
			return true;
		}
		return false;
	}

	virtual bool searched(uint64_t _startNonce, uint32_t _count)
	{
		Guard l(x_all);
		total += _count;
		last = _startNonce + _count;
		if (abort)
		{
			aborted = true;
			return true;
		}
		return false;
	}

	vector<Nonce> fetchFound() { vector<Nonce> ret; Guard l(x_all); std::swap(ret, found); return ret; }
	uint64_t fetchTotal() { Guard l(x_all); auto ret = total; total = 0; return ret; }

	Mutex x_all;
	vector<Nonce> found;
	uint64_t total;
	uint64_t last;
	bool abort = false;
	bool aborted = false;
};

EthashCL::EthashCL():
	m_miner(new ethash_cl_miner),
	m_hook(new EthashCLHook)
{
}

EthashCL::~EthashCL()
{
	m_hook->abort = true;
	for (unsigned timeout = 0; timeout < 100 && !m_hook->aborted; ++timeout)
		std::this_thread::sleep_for(chrono::milliseconds(30));
	if (!m_hook->aborted)
		cwarn << "Couldn't abort. Abandoning OpenCL process.";
}

bool EthashCL::verify(BlockInfo const& _header)
{
	return Ethasher::verify(_header);
}

std::pair<MineInfo, Ethash::Proof> EthashCL::mine(BlockInfo const& _header, unsigned _msTimeout, bool, bool)
{
	if (m_lastHeader.seedHash() != _header.seedHash())
	{
		m_miner->init(Ethasher::params(_header), _header.seedHash().data());
		// TODO: reinit probably won't work when seed changes.
	}
	if (m_lastHeader != _header)
	{
		static std::random_device s_eng;
		uint64_t tryNonce = (uint64_t)(u64)(m_last = Nonce::random(s_eng));
		m_miner->search(_header.headerHash(WithoutNonce).data(), tryNonce, *m_hook);
	}
	m_lastHeader = _header;

	std::this_thread::sleep_for(chrono::milliseconds(_msTimeout));
	auto found = m_hook->fetchFound();
	if (!found.empty())
	{
		h256 mixHash; // ?????
		return std::make_pair(MineInfo{0.0, 1e99, 0, true}, EthashCL::Proof((Nonce)(u64)found[0], mixHash));
	}
	return std::make_pair(MineInfo{0.0, 1e99, 0, false}, EthashCL::Proof());
}

#endif

}
}
