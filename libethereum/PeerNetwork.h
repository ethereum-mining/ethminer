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
/** @file PeerNetwork.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "Common.h"

namespace eth
{

class BlockChain;
class TransactionQueue;

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


