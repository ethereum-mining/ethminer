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
#include <thread>
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
	auto tid = std::this_thread::get_id();
	static std::mt19937_64 s_eng((time(0) + *reinterpret_cast<unsigned*>(m_last.data()) + std::hash<decltype(tid)>()(tid)));
	uint64_t tryNonce = (uint64_t)(u64)(m_last = Nonce::random(s_eng));

	h256 boundary = u256((bigint(1) << 256) / _header.difficulty);
	ret.first.requirement = log2((double)(u256)boundary);

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
		h256 val(m.mine(tryNonce));
		best = std::min<double>(best, log2((double)(u256)val));
		if (val <= boundary)
		{
			ret.first.completed = true;
			assert(Ethasher::eval(_header, (Nonce)(u64)tryNonce).value == val);
			result.mixHash = m.lastMixHash();
			result.nonce = u64(tryNonce);
			BlockInfo test = _header;
			assignResult(result, test);
			assert(verify(test));
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
class ethash_cl_miner
{
public:
	struct search_hook
	{
		// reports progress, return true to abort
		virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
		virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
	};

	ethash_cl_miner();

	bool init(ethash_params const& params, const uint8_t seed[32], unsigned workgroup_size = 64);

	void hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count);
	void search(uint8_t const* header, uint64_t target, search_hook& hook);
};
*/

struct EthashCLHook: public ethash_cl_miner::search_hook
{
	void abort()
	{
		if (m_aborted)
			return;
		m_abort = true;
		for (unsigned timeout = 0; timeout < 100 && !m_aborted; ++timeout)
			std::this_thread::sleep_for(chrono::milliseconds(30));
		if (!m_aborted)
			cwarn << "Couldn't abort. Abandoning OpenCL process.";
		m_aborted = m_abort = false;
		m_found.clear();
	}

	vector<Nonce> fetchFound() { vector<Nonce> ret; Guard l(x_all); std::swap(ret, m_found); return ret; }
	uint64_t fetchTotal() { Guard l(x_all); auto ret = m_total; m_total = 0; return ret; }

protected:
	virtual bool found(uint64_t const* _nonces, uint32_t _count) override
	{
		Guard l(x_all);
		for (unsigned i = 0; i < _count; ++i)
			m_found.push_back((Nonce)(u64)_nonces[i]);
		m_aborted = true;
		return true;
	}

	virtual bool searched(uint64_t _startNonce, uint32_t _count) override
	{
		Guard l(x_all);
		m_total += _count;
		m_last = _startNonce + _count;
		if (m_abort)
		{
			m_aborted = true;
			return true;
		}
		return false;
	}

private:
	Mutex x_all;
	vector<Nonce> m_found;
	uint64_t m_total;
	uint64_t m_last;
	bool m_abort = false;
	bool m_aborted = true;
};

EthashCL::EthashCL():
	m_hook(new EthashCLHook)
{
}

EthashCL::~EthashCL()
{
}

bool EthashCL::verify(BlockInfo const& _header)
{
	return Ethasher::verify(_header);
}

std::pair<MineInfo, Ethash::Proof> EthashCL::mine(BlockInfo const& _header, unsigned _msTimeout, bool, bool)
{
	if (!m_lastHeader || m_lastHeader.seedHash() != _header.seedHash())
	{
		if (m_miner)
			m_hook->abort();
		m_miner.reset(new ethash_cl_miner);
		m_miner->init(Ethasher::params(_header), [&](void* d){ Ethasher::get()->readFull(_header, d); });
	}
	if (m_lastHeader != _header)
	{
		m_hook->abort();
		static std::random_device s_eng;
		uint64_t tryNonce = (uint64_t)(u64)(m_last = Nonce::random(s_eng));
		m_miner->search(_header.headerHash(WithoutNonce).data(), tryNonce, *m_hook);
	}
	m_lastHeader = _header;

	std::this_thread::sleep_for(chrono::milliseconds(_msTimeout));
	auto found = m_hook->fetchFound();
	if (!found.empty())
	{
		Nonce n = (Nonce)(u64)found[0];
		auto result = Ethasher::eval(_header, n);
		return std::make_pair(MineInfo(true), EthashCL::Proof{n, result.mixHash});
	}
	return std::make_pair(MineInfo(false), EthashCL::Proof());
}

#endif

}
}
