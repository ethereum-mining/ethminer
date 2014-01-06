#include "Common.h"
#include "PeerNetwork.h"
using namespace std;
using namespace eth;

PeerNetwork::PeerNetwork()
{
}

PeerNetwork::~PeerNetwork()
{
}

void PeerNetwork::process()
{
}

void PeerNetwork::sync(BlockChain& _bc, TransactionQueue const& _tq)
{
/*
	while (incomingBlock())
	{
		// import new block
		bytes const& block = net.incomingBlock();
		_bc.import(block);
		net.popIncomingBlock();

		// check block chain and make longest given all available blocks.
		bc.rejig();
	}

	while (incomingTransaction())
	{
		bytes const& tx = net.incomingTransaction();
		_tq.import(tx);
		net.popIncomingTransaction();
	}
*/
}

