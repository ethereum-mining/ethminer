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
/** @file EthashCPUMiner.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#include "EthashCPUMiner.h"
#include <thread>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <random>
using namespace std;
using namespace dev;
using namespace eth;

unsigned EthashCPUMiner::s_numInstances = 0;

EthashCPUMiner::EthashCPUMiner(GenericMiner<EthashProofOfWork>::ConstructionInfo const& _ci):
	GenericMiner<EthashProofOfWork>(_ci), Worker("miner" + toString(index()))
{
}

EthashCPUMiner::~EthashCPUMiner()
{
}

void EthashCPUMiner::kickOff()
{
	stopWorking();
	startWorking();
}

void EthashCPUMiner::pause()
{
	stopWorking();
}

void EthashCPUMiner::workLoop()
{
	auto tid = std::this_thread::get_id();
	static std::mt19937_64 s_eng((time(0) + std::hash<decltype(tid)>()(tid)));

	uint64_t tryNonce = s_eng();
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

std::string EthashCPUMiner::platformInfo()
{
	return toString(std::thread::hardware_concurrency()) + "-thread CPU";
}
