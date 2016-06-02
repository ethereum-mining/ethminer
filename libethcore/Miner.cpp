#include "Miner.h"
#include "EthashAux.h"

using namespace dev;
using namespace eth;

unsigned dev::eth::GenericMiner<dev::eth::EthashProofOfWork>::s_dagLoadMode = 0;
unsigned dev::eth::GenericMiner<dev::eth::EthashProofOfWork>::s_dagLoadIndex = 0;


