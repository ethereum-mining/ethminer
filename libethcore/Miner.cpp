#include "Miner.h"
#include "EthashAux.h"

using namespace dev;
using namespace eth;

template <>
unsigned dev::eth::GenericMiner<dev::eth::EthashProofOfWork>::s_dagLoadMode = 0;

template <>
unsigned dev::eth::GenericMiner<dev::eth::EthashProofOfWork>::s_dagLoadIndex = 0;


