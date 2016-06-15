#include "Miner.h"
#include "EthashAux.h"

using namespace dev;
using namespace eth;

template <>
unsigned dev::eth::GenericMiner<dev::eth::EthashProofOfWork>::s_dagLoadMode = 0;

template <>
volatile unsigned dev::eth::GenericMiner<dev::eth::EthashProofOfWork>::s_dagLoadIndex = 0;

template <>
unsigned dev::eth::GenericMiner<dev::eth::EthashProofOfWork>::s_dagCreateDevice = 0;

template <>
volatile void* dev::eth::GenericMiner<dev::eth::EthashProofOfWork>::s_dagInHostMemory = NULL;


