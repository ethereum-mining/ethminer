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
/** @file WebThree.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "WebThree.h"

#include <chrono>
#include <thread>
#include <boost/filesystem.hpp>
#include <libdevcore/Log.h>
#include <libp2p/Host.h>
#include <libethereum/Defaults.h>
#include <libethereum/EthereumHost.h>
#include <libwhisper/WhisperPeer.h>
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::eth;
using namespace dev::shh;

WebThreeDirect::WebThreeDirect(std::string const& _clientVersion, std::string const& _dbPath, bool _forceClean, std::set<std::string> const& _interfaces, unsigned short _listenPort, std::string const& _publicIP, bool _upnp, dev::u256 _networkId, bool _localNetworking):
	m_clientVersion(_clientVersion),
	m_net(m_clientVersion, _listenPort, _publicIP, _upnp, _localNetworking)
{
	if (_dbPath.size())
		Defaults::setDBPath(_dbPath);

	if (_interfaces.count("eth"))
		m_ethereum.reset(new eth::Client(&m_net, _dbPath, _forceClean, _networkId));

//	if (_interfaces.count("shh"))
//		m_whisper = new eth::Whisper(m_net.get());

	static const char* c_threadName = "net";

	UpgradableGuard l(x_work);
	{
		UpgradeGuard ul(l);

		if (!m_work)
			m_work.reset(new thread([&]()
			{
				setThreadName(c_threadName);
				m_workState.store(Active, std::memory_order_release);
				while (m_workState.load(std::memory_order_acquire) != Deleting)
					workNet();
				m_workState.store(Deleted, std::memory_order_release);
			}));

	}
}

WebThreeDirect::~WebThreeDirect()
{
	UpgradableGuard l(x_work);

	if (m_work)
	{
		if (m_workState.load(std::memory_order_acquire) == Active)
			m_workState.store(Deleting, std::memory_order_release);
		while (m_workState.load(std::memory_order_acquire) != Deleted)
			this_thread::sleep_for(chrono::milliseconds(10));
		m_work->join();
	}
	if (m_work)
	{
		UpgradeGuard ul(l);
		m_work.reset(nullptr);
	}
}

std::vector<PeerInfo> WebThreeDirect::peers()
{
	ReadGuard l(x_work);
	return m_net.peers();
}

size_t WebThreeDirect::peerCount() const
{
	ReadGuard l(x_work);
	return m_net.peerCount();
}

void WebThreeDirect::setIdealPeerCount(size_t _n)
{
	ReadGuard l(x_work);
	return m_net.setIdealPeerCount(_n);
}

bytes WebThreeDirect::savePeers()
{
	ReadGuard l(x_work);
	return m_net.savePeers();
}

void WebThreeDirect::restorePeers(bytesConstRef _saved)
{
	ReadGuard l(x_work);
	return m_net.restorePeers(_saved);
}

void WebThreeDirect::connect(std::string const& _seedHost, unsigned short _port)
{
	ReadGuard l(x_work);
	m_net.connect(_seedHost, _port);
}

void WebThreeDirect::workNet()
{
	// Process network events.
	// Synchronise block chain with network.
	// Will broadcast any of our (new) transactions and blocks, and collect & add any of their (new) transactions and blocks.
	{
		ReadGuard l(x_work);
		m_net.process();	// must be in guard for now since it uses the blockchain.
	}
	this_thread::sleep_for(chrono::milliseconds(1));
}

