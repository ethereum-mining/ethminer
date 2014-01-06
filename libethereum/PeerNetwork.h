#pragma once

#include "Common.h"

namespace eth
{

class BlockChain;
class TransactionQueue;

static const bytes NullBytes;

class PeerNetwork
{
public:
	PeerNetwork();
	~PeerNetwork();

	/// Conduct I/O, polling, syncing, whatever.
	/// Ideally all time-consuming I/O is done in a background thread, but you get this call every 100ms or so anyway.
	void process();

	/// Sync with the BlockChain. It might contain one of our mined blocks, we might have new candidates from the network.
	void sync(BlockChain& _bc, TransactionQueue const&);
	
	/// Get an incoming transaction from the queue. @returns bytes() if nothing waiting.
	bytes const& incomingTransaction() { return NullBytes; }

	/// Remove incoming transaction from the queue. Make sure you've finished with the data from any previous incomingTransaction() calls.
	void popIncomingTransaction() {}

private:
};

}


