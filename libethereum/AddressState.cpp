#include "Trie.h"
#include "AddressState.h"
using namespace std;
using namespace eth;

u256 AddressState::memoryHash() const
{
	return hash256(m_memory);
}

