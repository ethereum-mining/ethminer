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
#include "Dagger.h"
#include "PeerNetwork.h"

namespace eth
{

struct MineProgress
{
	uint requirement;
	uint best;
	uint current;
};

class Client
{
public:
	Client(std::string const& _clientVersion, Address _us = Address(), std::string const& _dbPath = std::string());
	~Client();

	void transact(Secret _secret, Address _dest, u256 _amount, u256 _fee, u256s _data = u256s());

	void lock();
	void unlock();

	bool changed() const { auto ret = m_changed; m_changed = false; return ret; }

	State const& state() const { return m_s; }
	BlockChain const& blockChain() const { return m_bc; }
	TransactionQueue const& transactionQueue() const { return m_tq; }

	std::vector<PeerInfo> peers() { return m_net ? m_net->peers() : std::vector<PeerInfo>(); }
	unsigned peerCount() const { return m_net ? m_net->peerCount() : 0; }

	void startNetwork(short _listenPort = 30303, std::string const& _seedHost = std::string(), short _port = 30303);
	void connect(std::string const& _seedHost, short _port = 30303);
	void stopNetwork();

	void setAddress(Address _us) { m_s.setAddress(_us); }
	Address address() const { return m_s.address(); }
	void startMining();
	void stopMining();
	MineProgress miningProgress() const { return m_mineProgress; }

private:
	void work();

	std::string m_clientVersion;		///< Our end-application client's name/version.
	BlockChain m_bc;					///< Maintains block database.
	TransactionQueue m_tq;				///< Maintains list of incoming transactions not yet on the block chain.
	Overlay m_stateDB;					///< Acts as the central point for the state database, so multiple States can share it.
	State m_s;							///< The present state of the client.
	PeerServer* m_net = nullptr;		///< Should run in background and send us events when blocks found and allow us to send blocks as required.
	std::thread* m_work;				///< The work thread.
	std::mutex m_lock;
	enum { Active = 0, Deleting, Deleted } m_workState = Active;
	bool m_doMine = false;				///< Are we supposed to be mining?
	MineProgress m_mineProgress;

	mutable bool m_changed;
};

}
