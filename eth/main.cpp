/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Ethereum client.
 */

#include "PeerNetwork.h"
#include "BlockChain.h"
#include "State.h"
using namespace std;
using namespace eth;

int main()
{
	// Our address.
	h256 privkey = sha3("123");
	Address us = toPublic(privkey);	// TODO: should be loaded from config file/set at command-line.

	BlockChain bc;			// Maintains block database.
	TransactionQueue tq;	// Maintains list of incoming transactions not yet on the block chain.
	State s(us);

	// Synchronise the state according to the block chain - i.e. replay all transactions in block chain, in order.
	// In practise this won't need to be done since the State DB will contain the keys for the tries for most recent (and many old) blocks.
	// TODO: currently it contains keys for *all* blocks. Make it remove old ones.
	s.sync(bc);
	s.sync(tq);

	PeerNetwork net;		// TODO: Implement - should run in background and send us events when blocks found and allow us to send blocks as required.
	while (true)
	{
		// Process network events.
		net.process();

		// Synchronise block chain with network.
		// Will broadcast any of our (new) transactions and blocks, and collect & add any of their (new) transactions and blocks.
		net.sync(bc, tq);

		// Synchronise state to block chain.
		// This should remove any transactions on our queue that are included within our state.
		// It also guarantees that the state reflects the longest (valid!) chain on the block chain.
		//   This might mean reverting to an earlier state and replaying some blocks, or, (worst-case:
		//   if there are no checkpoints before our fork) reverting to the genesis block and replaying
		//   all blocks.
		s.sync(bc);		// Resynchronise state with block chain & trans
		s.sync(tq);

		// Mine for a while.
		if (s.mine(100))
		{
			// Mined block
			bytes b = s.blockData();

			// Import block.
			bc.import(b);
		}
	}

	return 0;
}
