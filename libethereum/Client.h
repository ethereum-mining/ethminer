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
/** @file Client.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <thread>
#include <mutex>
#include "Common.h"
#include "BlockChain.h"
#include "TransactionQueue.h"
#include "State.h"
#include "PeerNetwork.h"

namespace eth
{

class Client
{
public:
	Client(std::string const& _dbPath);
	~Client();

	void transact(Address _dest, u256 _amount, u256 _fee, u256s _data = u256s(), Secret _secret);

	BlockChain const& blockChain() const;
	TransactionQueue const& transactionQueue() const;

	unsigned peerCount() const;

	void startNetwork(short _listenPort = 30303, std::string const& _seedHost, short _port = 30303);
	void stopNetwork();

	void startMining();
	void stopMining();
	std::pair<unsigned, unsigned> miningProgress() const;

private:
	void work();

	BlockChain m_bc;					///< Maintains block database.
	TransactionQueue m_tq;				///< Maintains list of incoming transactions not yet on the block chain.
	Overlay m_stateDB;					///< Acts as the central point for the state database, so multiple States can share it.
	State m_s;							///< The present state of the client.
	PeerServer* m_net = nullptr;		///< Should run in background and send us events when blocks found and allow us to send blocks as required.
	std::thread* m_work;				///< The work thread.
	enum { Active = 0, Deleting, Deleted } m_workState = Active;
	bool m_doMine = false;				///< Are we supposed to be mining?
};

}
