#include "Miner.h"
#include "EthashAux.h"

using namespace dev;
using namespace eth;

unsigned dev::eth::Miner::s_dagLoadMode = 0;

volatile unsigned dev::eth::Miner::s_dagLoadIndex = 0;

unsigned dev::eth::Miner::s_dagCreateDevice = 0;

volatile void* dev::eth::Miner::s_dagInHostMemory = NULL;


