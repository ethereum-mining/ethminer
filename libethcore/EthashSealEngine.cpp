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
/** @file EthashSealEngine.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#include "EthashSealEngine.h"
#include "EthashCPUMiner.h"
#include "EthashGPUMiner.h"
using namespace std;
using namespace dev;
using namespace eth;

EthashSealEngine::EthashSealEngine()
{
	map<string, GenericFarm<EthashProofOfWork>::SealerDescriptor> sealers;
	sealers["cpu"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{&EthashCPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCPUMiner(ci); }};
#if ETH_ETHASHCL
	sealers["opencl"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{&EthashGPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashGPUMiner(ci); }};
#endif
	m_farm.setSealers(sealers);
}

strings EthashSealEngine::sealers() const
{
	return {
		"cpu"
#if ETH_ETHASHCL
		, "opencl"
#endif
	};
}

void EthashSealEngine::generateSeal(BlockInfo const& _bi)
{
	m_sealing = Ethash::BlockHeader(_bi);
	m_farm.setWork(m_sealing);
	m_farm.start(m_sealer);
	m_farm.setWork(m_sealing);		// TODO: take out one before or one after...
	bytes shouldPrecompute = option("precomputeDAG");
	if (!shouldPrecompute.empty() && shouldPrecompute[0] == 1)
		Ethash::ensurePrecomputed((unsigned)_bi.number());
}

void EthashSealEngine::onSealGenerated(std::function<void(bytes const&)> const& _f)
{
	m_farm.onSolutionFound([=](EthashProofOfWork::Solution const& sol)
	{
//		cdebug << m_farm.work().seedHash << m_farm.work().headerHash << sol.nonce << EthashAux::eval(m_farm.work().seedHash, m_farm.work().headerHash, sol.nonce).value;
		m_sealing.m_mixHash = sol.mixHash;
		m_sealing.m_nonce = sol.nonce;
		if (!m_sealing.preVerify())
			return false;
		RLPStream ret;
		m_sealing.streamRLP(ret);
		_f(ret.out());
		return true;
	});
}
