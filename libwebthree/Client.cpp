/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file RawWebThree.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Client.h"

#include <chrono>
#include <thread>
#include <boost/filesystem.hpp>
#include <libethential/Log.h>
#include <libp2p/Host.h>
#include <libethereum/Defaults.h>
#include <libethereum/EthereumHost.h>
#include <libwhisper/WhisperPeer.h>
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::eth;
using namespace dev::shh;

RawWebThree::RawWebThree(std::string const& _clientVersion, std::string const& _dbPath, bool _forceClean):
	m_clientVersion(_clientVersion)
{
	if (_dbPath.size())
		Defaults::setDBPath(_dbPath);
}

RawWebThree::~RawWebThree()
{
	stopNetwork();
}

void RawWebThree::startNetwork(unsigned short _listenPort, std::string const& _seedHost, unsigned short _port, NodeMode _mode, unsigned _peers, string const& _publicIP, bool _upnp, u256 _networkId)
{
	static const char* c_threadName = "net";

	{
		UpgradableGuard l(x_net);
		if (m_net.get())
			return;
		{
			UpgradeGuard ul(l);

			if (!m_workNet)
				m_workNet.reset(new thread([&]()
				{
					setThreadName(c_threadName);
					m_workNetState.store(Active, std::memory_order_release);
					while (m_workNetState.load(std::memory_order_acquire) != Deleting)
						workNet();
					m_workNetState.store(Deleted, std::memory_order_release);
				}));

			try
			{
				m_net.reset(new Host(m_clientVersion, _listenPort, _publicIP, _upnp));
			}
			catch (std::exception const&)
			{
				// Probably already have the port open.
				cwarn << "Could not initialize with specified/default port. Trying system-assigned port";
				m_net.reset(new Host(m_clientVersion, 0, _publicIP, _upnp));
			}
/*			if (_mode == NodeMode::Full)
				m_net->registerCapability(new EthereumHost(m_bc, _networkId));
			if (_mode == NodeMode::Full)
				m_net->registerCapability(new WhisperHost());*/
		}
		m_net->setIdealPeerCount(_peers);
	}

	if (_seedHost.size())
		connect(_seedHost, _port);
}

void RawWebThree::stopNetwork()
{
	UpgradableGuard l(x_net);

	if (m_workNet)
	{
		if (m_workNetState.load(std::memory_order_acquire) == Active)
			m_workNetState.store(Deleting, std::memory_order_release);
		while (m_workNetState.load(std::memory_order_acquire) != Deleted)
			this_thread::sleep_for(chrono::milliseconds(10));
		m_workNet->join();
	}
	if (m_net)
	{
		UpgradeGuard ul(l);
		m_net.reset(nullptr);
		m_workNet.reset(nullptr);
	}
}

std::vector<PeerInfo> RawWebThree::peers()
{
	ReadGuard l(x_net);
	return m_net ? m_net->peers() : std::vector<PeerInfo>();
}

size_t RawWebThree::peerCount() const
{
	ReadGuard l(x_net);
	return m_net ? m_net->peerCount() : 0;
}

void RawWebThree::setIdealPeerCount(size_t _n) const
{
	ReadGuard l(x_net);
	if (m_net)
		return m_net->setIdealPeerCount(_n);
}

bytes RawWebThree::savePeers()
{
	ReadGuard l(x_net);
	if (m_net)
		return m_net->savePeers();
	return bytes();
}

void RawWebThree::restorePeers(bytesConstRef _saved)
{
	ReadGuard l(x_net);
	if (m_net)
		return m_net->restorePeers(_saved);
}

void RawWebThree::connect(std::string const& _seedHost, unsigned short _port)
{
	ReadGuard l(x_net);
	if (!m_net.get())
		return;
	m_net->connect(_seedHost, _port);
}

void RawWebThree::workNet()
{
	// Process network events.
	// Synchronise block chain with network.
	// Will broadcast any of our (new) transactions and blocks, and collect & add any of their (new) transactions and blocks.
	{
		ReadGuard l(x_net);
		if (m_net)
		{
			m_net->process();	// must be in guard for now since it uses the blockchain.

			// returns h256Set as block hashes, once for each block that has come in/gone out.
//			m_net->cap<EthereumHost>()->sync(m_tq, m_bq);
		}
	}
	this_thread::sleep_for(chrono::milliseconds(1));
}

