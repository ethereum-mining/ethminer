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
#include <libdevcore/CommonJS.h>
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
#include "Exceptions.h"
#include "Farm.h"
#include "Miner.h"
#include "Params.h"
using namespace std;
using namespace std::chrono;

namespace dev
{
namespace eth
{

h256 const& Ethash::BlockHeaderRaw::seedHash() const
{
	if (!m_seedHash)
		m_seedHash = EthashAux::seedHash((unsigned)m_number);
	return m_seedHash;
}

void Ethash::BlockHeaderRaw::populateFromHeader(RLP const& _header, Strictness _s)
{
	m_mixHash = _header[BlockInfo::BasicFields].toHash<h256>();
	m_nonce = _header[BlockInfo::BasicFields + 1].toHash<h64>();

	// check it hashes according to proof of work or that it's the genesis block.
	if (_s == CheckEverything && m_parentHash && !verify())
	{
		InvalidBlockNonce ex;
		ex << errinfo_nonce(m_nonce);
		ex << errinfo_mixHash(m_mixHash);
		ex << errinfo_seedHash(seedHash());
		EthashProofOfWork::Result er = EthashAux::eval(seedHash(), hashWithout(), m_nonce);
		ex << errinfo_ethashResult(make_tuple(er.value, er.mixHash));
		ex << errinfo_hash256(hashWithout());
		ex << errinfo_difficulty(m_difficulty);
		ex << errinfo_target(boundary());
		BOOST_THROW_EXCEPTION(ex);
	}
	else if (_s == QuickNonce && m_parentHash && !preVerify())
	{
		InvalidBlockNonce ex;
		ex << errinfo_hash256(hashWithout());
		ex << errinfo_difficulty(m_difficulty);
		ex << errinfo_nonce(m_nonce);
		BOOST_THROW_EXCEPTION(ex);
	}

	if (_s != CheckNothing)
	{
		if (m_difficulty < c_minimumDifficulty)
			BOOST_THROW_EXCEPTION(InvalidDifficulty() << RequirementError(bigint(c_minimumDifficulty), bigint(m_difficulty)) );

		if (m_gasLimit < c_minGasLimit)
			BOOST_THROW_EXCEPTION(InvalidGasLimit() << RequirementError(bigint(c_minGasLimit), bigint(m_gasLimit)) );

		if (m_number && m_extraData.size() > c_maximumExtraDataSize)
			BOOST_THROW_EXCEPTION(ExtraDataTooBig() << RequirementError(bigint(c_maximumExtraDataSize), bigint(m_extraData.size())));
	}
}

void Ethash::BlockHeaderRaw::verifyParent(BlockHeaderRaw const& _parent)
{
	// Check difficulty is correct given the two timestamps.
	if (m_difficulty != calculateDifficulty(_parent))
		BOOST_THROW_EXCEPTION(InvalidDifficulty() << RequirementError((bigint)calculateDifficulty(_parent), (bigint)m_difficulty));

	if (m_gasLimit < c_minGasLimit ||
		m_gasLimit <= _parent.m_gasLimit - _parent.m_gasLimit / c_gasLimitBoundDivisor ||
		m_gasLimit >= _parent.m_gasLimit + _parent.m_gasLimit / c_gasLimitBoundDivisor)
		BOOST_THROW_EXCEPTION(InvalidGasLimit() << errinfo_min((bigint)_parent.m_gasLimit - _parent.m_gasLimit / c_gasLimitBoundDivisor) << errinfo_got((bigint)m_gasLimit) << errinfo_max((bigint)_parent.m_gasLimit + _parent.m_gasLimit / c_gasLimitBoundDivisor));
}

void Ethash::BlockHeaderRaw::populateFromParent(BlockHeaderRaw const& _parent)
{
	(void)_parent;
}

bool Ethash::BlockHeaderRaw::preVerify() const
{
	if (m_number >= ETHASH_EPOCH_LENGTH * 2048)
		return false;

	bool ret = !!ethash_quick_check_difficulty(
			(ethash_h256_t const*)hashWithout().data(),
			(uint64_t)(u64)m_nonce,
			(ethash_h256_t const*)m_mixHash.data(),
			(ethash_h256_t const*)boundary().data());
	return ret;
}

bool Ethash::BlockHeaderRaw::verify() const
{
	bool pre = preVerify();
#if !ETH_DEBUG
	if (!pre)
	{
		cwarn << "Fail on preVerify";
		return false;
	}
#endif

	auto result = EthashAux::eval(seedHash(), hashWithout(), m_nonce);
	bool slow = result.value <= boundary() && result.mixHash == m_mixHash;

//	cdebug << (slow ? "VERIFY" : "VERYBAD");
//	cdebug << result.value.hex() << _header.boundary().hex();
//	cdebug << result.mixHash.hex() << _header.mixHash.hex();

#if ETH_DEBUG || !ETH_TRUE
	if (!pre && slow)
	{
		cwarn << "WARNING: evaluated result gives true whereas ethash_quick_check_difficulty gives false.";
		cwarn << "headerHash:" << hashWithout();
		cwarn << "nonce:" << m_nonce;
		cwarn << "mixHash:" << m_mixHash;
		cwarn << "difficulty:" << m_difficulty;
		cwarn << "boundary:" << boundary();
		cwarn << "result.value:" << result.value;
		cwarn << "result.mixHash:" << result.mixHash;
	}
#endif

	return slow;
}

void Ethash::BlockHeaderRaw::prep(std::function<int(unsigned)> const& _f) const
{
	EthashAux::full(seedHash(), true, _f);
}

StringHashMap Ethash::BlockHeaderRaw::jsInfo() const
{
	return { { "nonce", toJS(m_nonce) }, { "seedHash", toJS(seedHash()) }, { "mixHash", toJS(m_mixHash) } };
}







class EthashCPUMiner: public GenericMiner<EthashProofOfWork>, Worker
{
public:
	EthashCPUMiner(GenericMiner<EthashProofOfWork>::ConstructionInfo const& _ci): GenericMiner<EthashProofOfWork>(_ci), Worker("miner" + toString(index())) {}

	static unsigned instances() { return s_numInstances > 0 ? s_numInstances : std::thread::hardware_concurrency(); }
	static std::string platformInfo();
	static void listDevices() {}
	static bool configureGPU(unsigned, unsigned, unsigned, unsigned, unsigned, bool, unsigned, uint64_t) { return false; }
	static void setNumInstances(unsigned _instances) { s_numInstances = std::min<unsigned>(_instances, std::thread::hardware_concurrency()); }

protected:
	void kickOff() override
	{
		stopWorking();
		startWorking();
	}

	void pause() override { stopWorking(); }

private:
	void workLoop() override;
	static unsigned s_numInstances;
};

#if ETH_ETHASHCL || !ETH_TRUE
class EthashGPUMiner: public GenericMiner<EthashProofOfWork>, Worker
{
	friend class dev::eth::EthashCLHook;

public:
	EthashGPUMiner(ConstructionInfo const& _ci);
	~EthashGPUMiner();

	static unsigned instances() { return s_numInstances > 0 ? s_numInstances : 1; }
	static std::string platformInfo();
	static unsigned getNumDevices();
	static void listDevices();
	static bool configureGPU(
		unsigned _localWorkSize,
		unsigned _globalWorkSizeMultiplier,
		unsigned _msPerBatch,
		unsigned _platformId,
		unsigned _deviceId,
		bool _allowCPU,
		unsigned _extraGPUMemory,
		uint64_t _currentBlock
	);
	static void setNumInstances(unsigned _instances) { s_numInstances = std::min<unsigned>(_instances, getNumDevices()); }

protected:
	void kickOff() override;
	void pause() override;

private:
	void workLoop() override;
	bool report(uint64_t _nonce);

	using GenericMiner<EthashProofOfWork>::accumulateHashes;

	EthashCLHook* m_hook = nullptr;
	ethash_cl_miner* m_miner = nullptr;

	h256 m_minerSeed;		///< Last seed in m_miner
	static unsigned s_platformId;
	static unsigned s_deviceId;
	static unsigned s_numInstances;
};
#endif

struct EthashSealEngine: public SealEngineBase<Ethash>
{
	friend class Ethash;

public:
	EthashSealEngine()
	{
		map<string, GenericFarm<EthashProofOfWork>::SealerDescriptor> sealers;
		sealers["cpu"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{&EthashCPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCPUMiner(ci); }};
#if ETH_ETHASHCL
		sealers["opencl"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{&EthashGPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashGPUMiner(ci); }};
#endif
		m_farm.setSealers(sealers);
	}

	strings sealers() const override
	{
		return {
			"cpu"
#if ETH_ETHASHCL
			, "opencl"
#endif
		};
	}
	void setSealer(std::string const& _sealer) override { m_sealer = _sealer; }
	void cancelGeneration() override { m_farm.stop(); }
	void generateSeal(BlockInfo const& _bi) override
	{
		m_sealing = Ethash::BlockHeader(_bi);
		m_farm.setWork(m_sealing);
		m_farm.start(m_sealer);
		m_farm.setWork(m_sealing);		// TODO: take out one before or one after...
		Ethash::ensurePrecomputed((unsigned)_bi.number());
	}
	void onSealGenerated(std::function<void(bytes const&)> const& _f) override
	{
		m_farm.onSolutionFound([=](EthashProofOfWork::Solution const& sol)
		{
			cdebug << m_farm.work().seedHash << m_farm.work().headerHash << sol.nonce << EthashAux::eval(m_farm.work().seedHash, m_farm.work().headerHash, sol.nonce).value;
			m_sealing.m_mixHash = sol.mixHash;
			m_sealing.m_nonce = sol.nonce;
			RLPStream ret;
			m_sealing.streamRLP(ret);
			_f(ret.out());
			return true;
		});
	}

private:
	bool m_opencl = false;
	eth::GenericFarm<EthashProofOfWork> m_farm;
	std::string m_sealer = "cpu";
	Ethash::BlockHeader m_sealing;
};

void Ethash::manuallySubmitWork(SealEngineFace* _engine, h256 const& _mixHash, Nonce _nonce)
{
	if (EthashSealEngine* e = dynamic_cast<EthashSealEngine*>(_engine))
		// Go via the farm since the handler function object is stored as a local within the Farm's lambda.
		// Has the side effect of stopping local workers, which is good, as long as it only does it for
		// valid submissions.
		static_cast<GenericFarmFace<EthashProofOfWork>&>(e->m_farm).submitProof(EthashProofOfWork::Solution{_nonce, _mixHash}, nullptr);
}

bool Ethash::isWorking(SealEngineFace* _engine)
{
	if (EthashSealEngine* e = dynamic_cast<EthashSealEngine*>(_engine))
		return e->m_farm.isMining();
	return false;
}

WorkingProgress Ethash::workingProgress(SealEngineFace* _engine)
{
	if (EthashSealEngine* e = dynamic_cast<EthashSealEngine*>(_engine))
		return e->m_farm.miningProgress();
	return WorkingProgress();
}

SealEngineFace* Ethash::createSealEngine()
{
	return new EthashSealEngine;
}

std::string Ethash::name()
{
	return "Ethash";
}

unsigned Ethash::revision()
{
	return ETHASH_REVISION;
}

void Ethash::ensurePrecomputed(unsigned _number)
{
	if (_number % ETHASH_EPOCH_LENGTH > ETHASH_EPOCH_LENGTH * 9 / 10)
		// 90% of the way to the new epoch
		EthashAux::computeFull(EthashAux::seedHash(_number + ETHASH_EPOCH_LENGTH), true);
}

unsigned EthashCPUMiner::s_numInstances = 0;

void EthashCPUMiner::workLoop()
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
		if (value <= boundary && submitProof(EthashProofOfWork::Solution{(h64)(u64)tryNonce, h256((uint8_t*)&ethashReturn.mix_hash, h256::ConstructFromPointer)}))
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

std::string EthashCPUMiner::platformInfo()
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
	EthashCLHook(EthashGPUMiner* _owner): m_owner(_owner) {}
	EthashCLHook(EthashCLHook const&) = delete;

	void abort()
	{
		{
			UniqueGuard l(x_all);
			if (m_aborted)
				return;
//		cdebug << "Attempting to abort";

			m_abort = true;
		}
		// m_abort is true so now searched()/found() will return true to abort the search.
		// we hang around on this thread waiting for them to point out that they have aborted since
		// otherwise we may end up deleting this object prior to searched()/found() being called.
		m_aborted.wait(true);
//		for (unsigned timeout = 0; timeout < 100 && !m_aborted; ++timeout)
//			std::this_thread::sleep_for(chrono::milliseconds(30));
//		if (!m_aborted)
//			cwarn << "Couldn't abort. Abandoning OpenCL process.";
	}

	void reset()
	{
		UniqueGuard l(x_all);
		m_aborted = m_abort = false;
	}

protected:
	virtual bool found(uint64_t const* _nonces, uint32_t _count) override
	{
//		dev::operator <<(std::cerr << "Found nonces: ", vector<uint64_t>(_nonces, _nonces + _count)) << std::endl;
		for (uint32_t i = 0; i < _count; ++i)
			if (m_owner->report(_nonces[i]))
				return (m_aborted = true);
		return m_owner->shouldStop();
	}

	virtual bool searched(uint64_t _startNonce, uint32_t _count) override
	{
		UniqueGuard l(x_all);
//		std::cerr << "Searched " << _count << " from " << _startNonce << std::endl;
		m_owner->accumulateHashes(_count);
		m_last = _startNonce + _count;
		if (m_abort || m_owner->shouldStop())
			return (m_aborted = true);
		return false;
	}

private:
	Mutex x_all;
	uint64_t m_last;
	bool m_abort = false;
	Notified<bool> m_aborted = {true};
	EthashGPUMiner* m_owner = nullptr;
};

unsigned EthashGPUMiner::s_platformId = 0;
unsigned EthashGPUMiner::s_deviceId = 0;
unsigned EthashGPUMiner::s_numInstances = 0;

EthashGPUMiner::EthashGPUMiner(ConstructionInfo const& _ci):
	GenericMiner<EthashProofOfWork>(_ci),
	Worker("gpuminer" + toString(index())),
	m_hook(new EthashCLHook(this))
{
}

EthashGPUMiner::~EthashGPUMiner()
{
	pause();
	delete m_miner;
	delete m_hook;
}

bool EthashGPUMiner::report(uint64_t _nonce)
{
	Nonce n = (Nonce)(u64)_nonce;
	EthashProofOfWork::Result r = EthashAux::eval(work().seedHash, work().headerHash, n);
	if (r.value < work().boundary)
		return submitProof(Solution{n, r.mixHash});
	return false;
}

void EthashGPUMiner::kickOff()
{
	m_hook->reset();
	startWorking();
}

void EthashGPUMiner::workLoop()
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
				if ((dag = EthashAux::full(w.seedHash, true)))
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
			m_miner->init(dagData.data(), dagData.size(), s_platformId, device);
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

void EthashGPUMiner::pause()
{
	m_hook->abort();
	stopWorking();
}

std::string EthashGPUMiner::platformInfo()
{
	return ethash_cl_miner::platform_info(s_platformId, s_deviceId);
}

unsigned EthashGPUMiner::getNumDevices()
{
	return ethash_cl_miner::getNumDevices(s_platformId);
}

void EthashGPUMiner::listDevices()
{
	return ethash_cl_miner::listDevices();
}

bool EthashGPUMiner::configureGPU(
	unsigned _localWorkSize,
	unsigned _globalWorkSizeMultiplier,
	unsigned _msPerBatch,
	unsigned _platformId,
	unsigned _deviceId,
	bool _allowCPU,
	unsigned _extraGPUMemory,
	uint64_t _currentBlock
)
{
	s_platformId = _platformId;
	s_deviceId = _deviceId;

	if (_localWorkSize != 32 && _localWorkSize != 64 && _localWorkSize != 128)
	{
		cout << "Given localWorkSize of " << toString(_localWorkSize) << "is invalid. Must be either 32,64, or 128" << endl;
		return false;
	}
	
	if (!ethash_cl_miner::configureGPU(
			_platformId,
			_localWorkSize,
			_globalWorkSizeMultiplier * _localWorkSize,
			_msPerBatch,
			_allowCPU,
			_extraGPUMemory,
			_currentBlock)
	)
	{
		cout << "No GPU device with sufficient memory was found. Can't GPU mine. Remove the -G argument" << endl;
		return false;
	}
	return true;
}

#endif

}
}
