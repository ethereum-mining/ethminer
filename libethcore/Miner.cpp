#include "Miner.h"
#include "EthashAux.h"

using namespace dev;
using namespace eth;

unsigned dev::eth::GenericMiner::s_dagLoadMode = 0;

volatile unsigned dev::eth::GenericMiner::s_dagLoadIndex = 0;

unsigned dev::eth::GenericMiner::s_dagCreateDevice = 0;

volatile void* dev::eth::GenericMiner::s_dagInHostMemory = NULL;


