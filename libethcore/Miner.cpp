#include "Miner.h"
#include "EthashAux.h"

using namespace dev;
using namespace eth;

unsigned dev::eth::Miner::s_dagLoadMode = 0;

volatile unsigned dev::eth::Miner::s_dagLoadIndex = 0;

unsigned dev::eth::Miner::s_dagCreateDevice = 0;

volatile void* dev::eth::Miner::s_dagInHostMemory = NULL;

namespace dev
{
namespace eth
{

bool Miner::report(uint64_t _nonce)
{
	WorkPackage w = work();  // Copy work package to avoid repeated mutex lock.
	Result r = EthashAux::eval(w.seedHash, w.headerHash, _nonce);
	if (r.value >= w.boundary)
		return false;

	if (m_farm.submitProof(Solution{_nonce, r.mixHash, w.headerHash, w.seedHash, w.boundary}, this))
	{
		// TODO: Even if the proof submitted, should be reset the work
		// package here and stop mining?
		Guard l(x_work);
		m_work.reset();
		return true;
	}
	return false;
}

}
}


