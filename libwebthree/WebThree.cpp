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
#include <boost/algorithm/string.hpp>

#include <libdevcore/Log.h>
#include <libethereum/Defaults.h>
#include <libethereum/EthereumHost.h>
#include <libwhisper/WhisperHost.h>
#include "BuildInfo.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::eth;
using namespace dev::shh;

WebThreeDirect::WebThreeDirect(
	std::string const& _clientVersion,
	std::string const& _dbPath,
	WithExisting _we,
	std::set<std::string> const& _interfaces,
	NetworkPreferences const& _n,
	bytesConstRef _network
):
	m_clientVersion(_clientVersion),
	m_net(_clientVersion, _n, _network)
{
	if (_dbPath.size())
		Defaults::setDBPath(_dbPath);
	if (_interfaces.count("eth"))
	{
		m_ethereum.reset(new eth::EthashClient(&m_net, shared_ptr<GasPricer>(), _dbPath, _we, 0));
		string bp = DEV_QUOTED(ETH_BUILD_PLATFORM);
		vector<string> bps;
		boost::split(bps, bp, boost::is_any_of("/"));
		bps[0] = bps[0].substr(0, 5);
		bps[1] = bps[1].substr(0, 3);
		bps.back() = bps.back().substr(0, 3);
		m_ethereum->setExtraData(rlpList(0, string(dev::Version) + "++" + string(DEV_QUOTED(ETH_COMMIT_HASH)).substr(0, 4) + (ETH_CLEAN_REPO ? "-" : "*") + string(DEV_QUOTED(ETH_BUILD_TYPE)).substr(0, 1) + boost::join(bps, "/")));
	}

	if (_interfaces.count("shh"))
		m_whisper = m_net.registerCapability<WhisperHost>(new WhisperHost);
}

WebThreeDirect::~WebThreeDirect()
{
	// Utterly horrible right now - WebThree owns everything (good), but:
	// m_net (Host) owns the eth::EthereumHost via a shared_ptr.
	// The eth::EthereumHost depends on eth::Client (it maintains a reference to the BlockChain field of Client).
	// eth::Client (owned by us via a unique_ptr) uses eth::EthereumHost (via a weak_ptr).
	// Really need to work out a clean way of organising ownership and guaranteeing startup/shutdown is perfect.

	// Have to call stop here to get the Host to kill its io_service otherwise we end up with left-over reads,
	// still referencing Sessions getting deleted *after* m_ethereum is reset, causing bad things to happen, since
	// the guarantee is that m_ethereum is only reset *after* all sessions have ended (sessions are allowed to
	// use bits of data owned by m_ethereum).
	m_net.stop();
	m_ethereum.reset();
}

std::string WebThreeDirect::composeClientVersion(std::string const& _client, std::string const& _clientName)
{
	return _client + "-" + "v" + dev::Version + "-" + string(DEV_QUOTED(ETH_COMMIT_HASH)).substr(0, 8) + (ETH_CLEAN_REPO ? "" : "*") + "/" + _clientName + "/" DEV_QUOTED(ETH_BUILD_TYPE) "-" DEV_QUOTED(ETH_BUILD_PLATFORM);
}

p2p::NetworkPreferences const& WebThreeDirect::networkPreferences() const
{
	return m_net.networkPreferences();
}

void WebThreeDirect::setNetworkPreferences(p2p::NetworkPreferences const& _n, bool _dropPeers)
{
	auto had = isNetworkStarted();
	if (had)
		stopNetwork();
	m_net.setNetworkPreferences(_n, _dropPeers);
	if (had)
		startNetwork();
}

std::vector<PeerSessionInfo> WebThreeDirect::peers()
{
	return m_net.peerSessionInfo();
}

size_t WebThreeDirect::peerCount() const
{
	return m_net.peerCount();
}

void WebThreeDirect::setIdealPeerCount(size_t _n)
{
	return m_net.setIdealPeerCount(_n);
}

bytes WebThreeDirect::saveNetwork()
{
	return m_net.saveNetwork();
}

void WebThreeDirect::addNode(NodeId const& _node, bi::tcp::endpoint const& _host)
{
	m_net.addNode(_node, NodeIPEndpoint(_host.address(), _host.port(), _host.port()));
}

void WebThreeDirect::requirePeer(NodeId const& _node, bi::tcp::endpoint const& _host)
{
	m_net.requirePeer(_node, NodeIPEndpoint(_host.address(), _host.port(), _host.port()));
}


