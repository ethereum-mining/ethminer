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
#include "BlockInfo.h"
#include "EthashAux.h"
#include "Exceptions.h"
#include "Farm.h"
#include "Miner.h"
#include "Params.h"
#include "EthashSealEngine.h"
#include "EthashCPUMiner.h"
#include "EthashGPUMiner.h"
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
			BOOST_THROW_EXCEPTION(ExtraDataTooBig() << RequirementError(bigint(c_maximumExtraDataSize), bigint(m_extraData.size())) << errinfo_extraData(m_extraData));
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

void Ethash::manuallySetWork(SealEngineFace* _engine, BlockHeader const& _work)
{
	// set m_sealing to the current problem.
	if (EthashSealEngine* e = dynamic_cast<EthashSealEngine*>(_engine))
		e->m_sealing = _work;
}

void Ethash::manuallySubmitWork(SealEngineFace* _engine, h256 const& _mixHash, Nonce _nonce)
{
	if (EthashSealEngine* e = dynamic_cast<EthashSealEngine*>(_engine))
	{
		// Go via the farm since the handler function object is stored as a local within the Farm's lambda.
		// Has the side effect of stopping local workers, which is good, as long as it only does it for
		// valid submissions.
		static_cast<GenericFarmFace<EthashProofOfWork>&>(e->m_farm).submitProof(EthashProofOfWork::Solution{_nonce, _mixHash}, nullptr);
	}
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

}
}
