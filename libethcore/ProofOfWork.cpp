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
#if ETH_ETHASHCL || !ETH_TRUE
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

void Ethash::CPUMiner::workLoop()
{
	Solution solution;

	class Miner
	{
	public:
		Miner(BlockInfo const& _header):
			m_headerHash(_header.headerHash(WithoutNonce)),
			m_params(Ethasher::params(_header)),
			m_datasetPointer(Ethasher::get()->full(_header).data())
		{}

		inline h256 mine(uint64_t _nonce)
		{
			ethash_compute_full(&m_ethashReturn, m_datasetPointer, &m_params, m_headerHash.data(), _nonce);
//			cdebug << "Ethasher::mine hh:" << m_headerHash << "nonce:" << (Nonce)(u64)_nonce << " => " << h256(m_ethashReturn.result, h256::ConstructFromPointer);
			return h256(m_ethashReturn.result, h256::ConstructFromPointer);
		}

		inline h256 lastMixHash() const
		{
			return h256(m_ethashReturn.mix_hash, h256::ConstructFromPointer);
		}

	private:
		ethash_return_value m_ethashReturn;
		h256 m_headerHash;
		ethash_params m_params;
		void const* m_datasetPointer;
	};

	Ethasher::Miner m(_header);

	std::pair<MineInfo, Solution> ret;
	auto tid = std::this_thread::get_id();
	static std::mt19937_64 s_eng((time(0) + *reinterpret_cast<unsigned*>(m_last.data()) + std::hash<decltype(tid)>()(tid)));
	uint64_t tryNonce = (uint64_t)(u64)(m_last = Nonce::random(s_eng));

	h256 boundary = _header.boundary();
	ret.first.requirement = log2((double)(u256)boundary);

	// 2^ 0      32      64      128      256
	//   [--------*-------------------------]
	//
	// evaluate until we run out of time
	auto startTime = std::chrono::steady_clock::now();
	double best = 1e99;	// high enough to be effectively infinity :)
	Solution result;
	unsigned hashCount = 0;
	for (; !shouldStop(); tryNonce++, hashCount++)
	{
		h256 val(m.mine(tryNonce));
		best = std::min<double>(best, log2((double)(u256)val));
		if (val <= boundary)
		{
			if (submitProof(solution))
				return;
		}
	}
	ret.first.hashes = hashCount;
	ret.first.best = best;
	ret.second = result;

	return;
}

#if ETH_ETHASHCL || !ETH_TRUE

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

namespace dev { namespace eth {
class EthashCLHook: public ethash_cl_miner::search_hook
{
public:
	EthashCLHook(Ethash::GPUMiner* _owner): m_owner(_owner) {}

	void abort()
	{
		Guard l(x_all);
		if (m_aborted)
			return;
//		cdebug << "Attempting to abort";
		m_abort = true;
		for (unsigned timeout = 0; timeout < 100 && !m_aborted; ++timeout)
			std::this_thread::sleep_for(chrono::milliseconds(30));
		if (!m_aborted)
			cwarn << "Couldn't abort. Abandoning OpenCL process.";
		m_aborted = m_abort = false;
	}

	uint64_t fetchTotal() { Guard l(x_all); auto ret = m_total; m_total = 0; return ret; }

protected:
	virtual bool found(uint64_t const* _nonces, uint32_t _count) override
	{
//		cdebug << "Found nonces: " << vector<uint64_t>(_nonces, _nonces + _count);
		for (uint32_t i = 0; i < _count; ++i)
		{
			if (m_owner->found(_nonces[i]))
			{
				m_aborted = true;
				return true;
			}
		}
		return false;
	}

	virtual bool searched(uint64_t _startNonce, uint32_t _count) override
	{
		Guard l(x_all);
//		cdebug << "Searched" << _count << "from" << _startNonce;
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
	uint64_t m_total;
	uint64_t m_last;
	bool m_abort = false;
	bool m_aborted = true;
	Ethash::GPUMiner* m_owner = nullptr;
};

} }

Ethash::GPUMiner::GPUMiner(ConstructionInfo const& _ci):
	Miner(_ci),
	m_hook(new EthashCLHook(this))
{
}

void Ethash::GPUMiner::report(uint64_t _nonce)
{
	Nonce n = (Nonce)(u64)_nonce;
	Ethasher::Result r = Ethasher::eval(m_work.seedHash, m_work.headerHash, n);
	if (r.value < m_work.boundary)
		return submitProof(Solution{n, r.mixHash});
	return false;
}

void Ethash::GPUMiner::kickOff(WorkPackage const& _work)
{
	if (!m_miner || m_minerSeed != _work.seedHash)
	{
		if (m_miner)
			m_hook->abort();
		m_miner.reset(new ethash_cl_miner);
		auto p = Ethasher::params(_work.seedHash);
		auto cb = [&](void* d) { Ethasher::get()->readFull(_work.seedHash, bytesRef((byte*)d, p.full_size)); };
		m_miner->init(p, cb, 32);
	}
	if (m_lastWork.headerHash != _work.headerHash)
	{
		m_hook->abort();
		uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)_work.boundary >> 192);
		m_miner->search(_work.headerHash, upper64OfBoundary, *m_hook);
	}
	m_work = _work;
}

void Ethash::GPUMiner::pause()
{
	m_hook->abort();
}

#endif

}
}
