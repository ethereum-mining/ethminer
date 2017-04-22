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
#include "EthashCUDAMiner.h"
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
#if ETH_ETHASHCUDA
	sealers["cuda"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{ &EthashCUDAMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCUDAMiner(ci); } };
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
#if ETH_ETHASHCUDA
		, "cuda"
#endif
	};
}
