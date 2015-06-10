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
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <array>
#include <thread>
#include <random>
#include <thread>
#include <libdevcore/Guards.h>
#include <libdevcore/Log.h>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libdevcrypto/CryptoPP.h>
#include <libdevcore/FileSystem.h>
#include <libethash/ethash.h>
#include <libethash/internal.h>
#if ETH_ETHASHCL || !ETH_TRUE
#include <libethash-cl/ethash_cl_miner.h>
#endif
#if ETH_CPUID || !ETH_TRUE
#define HAVE_STDINT_H
#include <libcpuid/libcpuid.h>
#endif
#include "BlockInfo.h"
#include "EthashAux.h"
using namespace std;
using namespace std::chrono;

namespace dev
{
namespace eth
{

const Ethash::WorkPackage Ethash::NullWorkPackage = Ethash::WorkPackage();

std::string Ethash::name()
{
	return "Ethash";
}

unsigned Ethash::revision()
{
	return ETHASH_REVISION;
}

Ethash::WorkPackage Ethash::package(BlockInfo const& _bi)
{
	WorkPackage ret;
	ret.boundary = _bi.boundary();
	ret.headerHash = _bi.headerHash(WithoutNonce);
	ret.seedHash = _bi.seedHash();
	return ret;
}

void Ethash::ensurePrecomputed(unsigned _number)
{
	if (_number % ETHASH_EPOCH_LENGTH > ETHASH_EPOCH_LENGTH * 9 / 10)
		// 90% of the way to the new epoch
		EthashAux::computeFull(EthashAux::seedHash(_number + ETHASH_EPOCH_LENGTH), true);
}

void Ethash::prep(BlockInfo const& _header, std::function<int(unsigned)> const& _f)
{
	EthashAux::full(_header.seedHash(), true, _f);
}

bool Ethash::preVerify(BlockInfo const& _header)
{
	if (_header.number >= ETHASH_EPOCH_LENGTH * 2048)
		return false;

	h256 boundary = u256((bigint(1) << 256) / _header.difficulty);

	bool ret = !!ethash_quick_check_difficulty(
			(ethash_h256_t const*)_header.headerHash(WithoutNonce).data(),
			(uint64_t)(u64)_header.nonce,
			(ethash_h256_t const*)_header.mixHash.data(),
			(ethash_h256_t const*)boundary.data());

	return ret;
}

bool Ethash::verify(BlockInfo const& _header)
{
	bool pre = preVerify(_header);
#if !ETH_DEBUG
	if (!pre)
		return false;
#endif

	auto result = EthashAux::eval(_header);
	bool slow = result.value <= _header.boundary() && result.mixHash == _header.mixHash;

//	cdebug << (slow ? "VERIFY" : "VERYBAD");
//	cdebug << result.value.hex() << _header.boundary().hex();
//	cdebug << result.mixHash.hex() << _header.mixHash.hex();

#if ETH_DEBUG || !ETH_TRUE
	if (!pre && slow)
	{
		cwarn << "WARNING: evaluated result gives true whereas ethash_quick_check_difficulty gives false.";
		cwarn << "headerHash:" << _header.headerHash(WithoutNonce);
		cwarn << "nonce:" << _header.nonce;
		cwarn << "mixHash:" << _header.mixHash;
		cwarn << "difficulty:" << _header.difficulty;
		cwarn << "boundary:" << _header.boundary();
		cwarn << "result.value:" << result.value;
		cwarn << "result.mixHash:" << result.mixHash;
	}
#endif

	return slow;
}

unsigned Ethash::CPUMiner::s_numInstances = 0;

void Ethash::CPUMiner::workLoop()
{
	auto tid = std::this_thread::get_id();
	static std::mt19937_64 s_eng((time(0) + std::hash<decltype(tid)>()(tid)));

	uint64_t tryNonce = (uint64_t)(u64)Nonce::random(s_eng);
	ethash_return_value ethashReturn;

	WorkPackage w = work();

	EthashAux::FullType dag;
	while (!shouldStop() && !dag)
	{
		while (!shouldStop() && EthashAux::computeFull(w.seedHash, true) != 100)
			this_thread::sleep_for(chrono::milliseconds(500));
		dag = EthashAux::full(w.seedHash, false);
	}

	h256 boundary = w.boundary;
	unsigned hashCount = 1;
	for (; !shouldStop(); tryNonce++, hashCount++)
	{
		ethashReturn = ethash_full_compute(dag->full, *(ethash_h256_t*)w.headerHash.data(), tryNonce);
		h256 value = h256((uint8_t*)&ethashReturn.result, h256::ConstructFromPointer);
		if (value <= boundary && submitProof(Solution{(Nonce)(u64)tryNonce, h256((uint8_t*)&ethashReturn.mix_hash, h256::ConstructFromPointer)}))
			break;
		if (!(hashCount % 100))
			accumulateHashes(100);
	}
}

static string jsonEncode(map<string, string> const& _m)
{
	string ret = "{";

	for (auto const& i: _m)
	{
		string k = boost::replace_all_copy(boost::replace_all_copy(i.first, "\\", "\\\\"), "'", "\\'");
		string v = boost::replace_all_copy(boost::replace_all_copy(i.second, "\\", "\\\\"), "'", "\\'");
		if (ret.size() > 1)
			ret += ", ";
		ret += "\"" + k + "\":\"" + v + "\"";
	}

	return ret + "}";
}

std::string Ethash::CPUMiner::platformInfo()
{
	string baseline = toString(std::thread::hardware_concurrency()) + "-thread CPU";
#if ETH_CPUID || !ETH_TRUE
	if (!cpuid_present())
		return baseline;
	struct cpu_raw_data_t raw;
	struct cpu_id_t data;
	if (cpuid_get_raw_data(&raw) < 0)
		return baseline;
	if (cpu_identify(&raw, &data) < 0)
		return baseline;
	map<string, string> m;
	m["vendor"] = data.vendor_str;
	m["codename"] = data.cpu_codename;
	m["brand"] = data.brand_str;
	m["L1 cache"] = toString(data.l1_data_cache);
	m["L2 cache"] = toString(data.l2_cache);
	m["L3 cache"] = toString(data.l3_cache);
	m["cores"] = toString(data.num_cores);
	m["threads"] = toString(data.num_logical_cpus);
	m["clocknominal"] = toString(cpu_clock_by_os());
	m["clocktested"] = toString(cpu_clock_measure(200, 0));
	/*
	printf("  MMX         : %s\n", data.flags[CPU_FEATURE_MMX] ? "present" : "absent");
	printf("  MMX-extended: %s\n", data.flags[CPU_FEATURE_MMXEXT] ? "present" : "absent");
	printf("  SSE         : %s\n", data.flags[CPU_FEATURE_SSE] ? "present" : "absent");
	printf("  SSE2        : %s\n", data.flags[CPU_FEATURE_SSE2] ? "present" : "absent");
	printf("  3DNow!      : %s\n", data.flags[CPU_FEATURE_3DNOW] ? "present" : "absent");
	*/
	return jsonEncode(m);
#else
	return baseline;
#endif
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
	}

	void reset()
	{
		m_aborted = m_abort = false;
	}

protected:
	virtual bool found(uint64_t const* _nonces, uint32_t _count) override
	{
//		dev::operator <<(std::cerr << "Found nonces: ", vector<uint64_t>(_nonces, _nonces + _count)) << std::endl;
		for (uint32_t i = 0; i < _count; ++i)
		{
			if (m_owner->report(_nonces[i]))
			{
				m_aborted = true;
				return true;
			}
		}
		return m_owner->shouldStop();
	}

	virtual bool searched(uint64_t _startNonce, uint32_t _count) override
	{
		Guard l(x_all);
//		std::cerr << "Searched " << _count << " from " << _startNonce << std::endl;
		m_owner->accumulateHashes(_count);
		m_last = _startNonce + _count;
		if (m_abort || m_owner->shouldStop())
		{
			m_aborted = true;
			return true;
		}
		return false;
	}

private:
	Mutex x_all;
	uint64_t m_last;
	bool m_abort = false;
	bool m_aborted = true;
	Ethash::GPUMiner* m_owner = nullptr;
};

unsigned Ethash::GPUMiner::s_platformId = 0;
unsigned Ethash::GPUMiner::s_deviceId = 0;
unsigned Ethash::GPUMiner::s_numInstances = 0;
unsigned Ethash::GPUMiner::s_dagChunks = 1;

Ethash::GPUMiner::GPUMiner(ConstructionInfo const& _ci):
	Miner(_ci),
	Worker("gpuminer" + toString(index())),
	m_hook(new EthashCLHook(this))
{
}

Ethash::GPUMiner::~GPUMiner()
{
	pause();
	delete m_miner;
	delete m_hook;
}

bool Ethash::GPUMiner::report(uint64_t _nonce)
{
	Nonce n = (Nonce)(u64)_nonce;
	Result r = EthashAux::eval(work().seedHash, work().headerHash, n);
	if (r.value < work().boundary)
		return submitProof(Solution{n, r.mixHash});
	return false;
}

void Ethash::GPUMiner::kickOff()
{
	m_hook->reset();
	startWorking();
}

void Ethash::GPUMiner::workLoop()
{
	// take local copy of work since it may end up being overwritten by kickOff/pause.
	try {
		WorkPackage w = work();
		cnote << "workLoop" << !!m_miner << m_minerSeed << w.seedHash;
		if (!m_miner || m_minerSeed != w.seedHash)
		{
			cnote << "Initialising miner...";
			m_minerSeed = w.seedHash;

			delete m_miner;
			m_miner = new ethash_cl_miner;

			unsigned device = instances() > 1 ? index() : s_deviceId;

			EthashAux::FullType dag;
			while (true)
			{
				if ((dag = EthashAux::full(w.seedHash, false)))
					break;
				if (shouldStop())
				{
					delete m_miner;
					m_miner = nullptr;
					return;
				}
				cnote << "Awaiting DAG";
				this_thread::sleep_for(chrono::milliseconds(500));
			}
			bytesConstRef dagData = dag->data();
			m_miner->init(dagData.data(), dagData.size(), 32, s_platformId, device, s_dagChunks);
		}

		uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)w.boundary >> 192);
		m_miner->search(w.headerHash.data(), upper64OfBoundary, *m_hook);
	}
	catch (cl::Error const& _e)
	{
		delete m_miner;
		m_miner = nullptr;
		cwarn << "Error GPU mining: " << _e.what() << "(" << _e.err() << ")";
	}
}

void Ethash::GPUMiner::pause()
{
	m_hook->abort();
	stopWorking();
}

std::string Ethash::GPUMiner::platformInfo()
{
	return ethash_cl_miner::platform_info(s_platformId, s_deviceId);
}

unsigned Ethash::GPUMiner::getNumDevices()
{
	return ethash_cl_miner::get_num_devices(s_platformId);
}

void Ethash::GPUMiner::listDevices()
{
	return ethash_cl_miner::listDevices();
}

bool Ethash::GPUMiner::haveSufficientMemory()
{
	return ethash_cl_miner::haveSufficientGPUMemory();
}

#endif

}
}
