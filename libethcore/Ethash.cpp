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
/** @file Ethash.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Ethash.h"

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
#include <libethash/ethash.h>
#if ETH_ETHASHCL || !ETH_TRUE
#include <libethash-cl/ethash_cl_miner.h>
#endif
#include "BlockInfo.h"
#include "EthashAux.h"
using namespace std;
using namespace std::chrono;

namespace dev
{
namespace eth
{

const Ethash::WorkPackage Ethash::NullWorkPackage;

std::string Ethash::name()
{
	return "Ethash";
}

unsigned Ethash::revision()
{
	return ETHASH_REVISION;
}

void Ethash::prep(BlockInfo const& _header)
{
	if (_header.number % ETHASH_EPOCH_LENGTH == 1)
		EthashAux::full(_header);
}

bool Ethash::preVerify(BlockInfo const& _header)
{
	if (_header.number >= ETHASH_EPOCH_LENGTH * 2048)
		return false;

	h256 boundary = u256((bigint(1) << 256) / _header.difficulty);

	return ethash_quick_check_difficulty(
		_header.headerHash(WithoutNonce).data(),
		(uint64_t)(u64)_header.nonce,
		_header.mixHash.data(),
		boundary.data());
}

bool Ethash::verify(BlockInfo const& _header)
{
	bool pre = preVerify(_header);
#if !ETH_DEBUG
	if (!pre)
		return false;
#endif

	h256 boundary = u256((bigint(1) << 256) / _header.difficulty);
	auto result = EthashAux::eval(_header);
	bool slow = result.value <= boundary && result.mixHash == _header.mixHash;

#if ETH_DEBUG || !ETH_TRUE
	if (!pre && slow)
	{
		cwarn << "WARNING: evaluated result gives true whereas ethash_quick_check_difficulty gives false.";
		cwarn << "headerHash:" << _header.headerHash(WithoutNonce);
		cwarn << "nonce:" << _header.nonce;
		cwarn << "mixHash:" << _header.mixHash;
		cwarn << "difficulty:" << _header.difficulty;
		cwarn << "boundary:" << boundary;
		cwarn << "result.value:" << result.value;
		cwarn << "result.mixHash:" << result.mixHash;
	}
#endif

	return slow;
}

void Ethash::CPUMiner::workLoop()
{
	auto tid = std::this_thread::get_id();
	static std::mt19937_64 s_eng((time(0) + std::hash<decltype(tid)>()(tid)));

	uint64_t tryNonce = (uint64_t)(u64)Nonce::random(s_eng);
	ethash_return_value ethashReturn;

	auto p = EthashAux::params(m_work.seedHash);
	void const* dagPointer = EthashAux::full(m_work.headerHash).data();
	uint8_t const* headerHashPointer = m_work.headerHash.data();
	h256 boundary = m_work.boundary;
	unsigned hashCount = 0;
	for (; !shouldStop(); tryNonce++, hashCount++)
	{
		ethash_compute_full(&ethashReturn, dagPointer, &p, headerHashPointer, tryNonce);
		h256 value = h256(ethashReturn.result, h256::ConstructFromPointer);
		if (value <= boundary && submitProof(Solution{(Nonce)(u64)tryNonce, h256(ethashReturn.mix_hash, h256::ConstructFromPointer)}))
			break;
	}
}

#if ETH_ETHASHCL || !ETH_TRUE

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
//		if (!m_aborted)
//			cwarn << "Couldn't abort. Abandoning OpenCL process.";
		m_aborted = m_abort = false;
	}

	uint64_t fetchTotal() { Guard l(x_all); auto ret = m_total; m_total = 0; return ret; }

protected:
	virtual bool found(uint64_t const* _nonces, uint32_t _count) override
	{
//		cdebug << "Found nonces: " << vector<uint64_t>(_nonces, _nonces + _count);
		for (uint32_t i = 0; i < _count; ++i)
		{
			if (m_owner->report(_nonces[i]))
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

Ethash::GPUMiner::GPUMiner(ConstructionInfo const& _ci):
	Miner(_ci),
	m_hook(new EthashCLHook(this))
{
}

bool Ethash::GPUMiner::report(uint64_t _nonce)
{
	Nonce n = (Nonce)(u64)_nonce;
	Result r = EthashAux::eval(m_lastWork.seedHash, m_lastWork.headerHash, n);
	if (r.value < m_lastWork.boundary)
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
		auto p = EthashAux::params(_work.seedHash);
		auto cb = [&](void* d) { EthashAux::full(_work.seedHash, bytesRef((byte*)d, p.full_size)); };
		m_miner->init(p, cb, 32);
	}
	if (m_lastWork.headerHash != _work.headerHash)
	{
		m_hook->abort();
		uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)_work.boundary >> 192);
		m_miner->search(_work.headerHash.data(), upper64OfBoundary, *m_hook);
	}
	m_lastWork = _work;
}

void Ethash::GPUMiner::pause()
{
	m_hook->abort();
}

#endif

}
}
