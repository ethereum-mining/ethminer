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
/** @file PeerNetwork.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

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

