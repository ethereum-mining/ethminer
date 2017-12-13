#include "Miner.h"
#include "EthashAux.h"

using namespace dev;
using namespace eth;

unsigned dev::eth::Miner::s_dagLoadMode = 0;

atomic<unsigned> dev::eth::Miner::s_dagLoadIndex(0);

unsigned dev::eth::Miner::s_dagCreateDevice = 0;

atomic<uint8_t*> dev::eth::Miner::s_dagInHostMemory(NULL);


