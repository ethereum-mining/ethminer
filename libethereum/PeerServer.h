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
/** @file PeerServer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <map>
#include <vector>
#include <set>
#include <memory>
#include <utility>
#include <thread>
#include <libethcore/CommonEth.h>
#include "PeerNetwork.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace eth
{

class PeerServer
{
	friend class PeerSession;

public:
	/// Start server, listening for connections on the given port.
	PeerServer(std::string const& _clientVersion, BlockChain const& _ch, unsigned int _networkId, unsigned short _port, NodeMode _m = NodeMode::Full, std::string const& _publicAddress = std::string(), bool _upnp = true);
	/// Start server, listening for connections on a system-assigned port.
	PeerServer(std::string const& _clientVersion, BlockChain const& _ch, unsigned int _networkId, NodeMode _m = NodeMode::Full, std::string const& _publicAddress = std::string(), bool _upnp = true);
	/// Start server, but don't listen.
	PeerServer(std::string const& _clientVersion, BlockChain const& _ch, unsigned int _networkId, NodeMode _m = NodeMode::Full);

	~PeerServer();

	static unsigned protocolVersion();
	unsigned networkId() { return m_networkId; }

	/// Connect to a peer explicitly.
	void connect(std::string const& _addr, unsigned short _port = 30303) noexcept;
	void connect(bi::tcp::endpoint const& _ep);

	/// Sync with the BlockChain. It might contain one of our mined blocks, we might have new candidates from the network.
	bool sync(BlockChain& _bc, TransactionQueue&, OverlayDB& _o);
	bool sync();

	/// Conduct I/O, polling, syncing, whatever.
	/// Ideally all time-consuming I/O is done in a background thread or otherwise asynchronously, but you get this call every 100ms or so anyway.
	/// This won't touch alter the blockchain.
	void process() { if (isInitialised()) m_ioService.poll(); }

	/// Set ideal number of peers.
	void setIdealPeerCount(unsigned _n) { m_idealPeerCount = _n; }

	void setMode(NodeMode _m) { m_mode = _m; }

	/// Get peer information.
    std::vector<PeerInfo> peers(bool _updatePing = false) const;

	/// Get number of peers connected; equivalent to, but faster than, peers().size().
	size_t peerCount() const { return m_peers.size(); }

	/// Ping the peers, to update the latency information.
	void pingAll();

	/// Get the port we're listening on currently.
	unsigned short listenPort() const { return m_public.port(); }

	bytes savePeers() const;
	void restorePeers(bytesConstRef _b);

private:
	void seal(bytes& _b);
	void populateAddresses();
	void determinePublic(std::string const& _publicAddress, bool _upnp);
	void ensureAccepting();

	///	Check to see if the network peer-state initialisation has happened.
	bool isInitialised() const { return m_latestBlockSent; }
	/// Initialises the network peer-state, doing the stuff that needs to be once-only. @returns true if it really was first.
	bool ensureInitialised(BlockChain& _bc, TransactionQueue& _tq);

	std::map<Public, bi::tcp::endpoint> potentialPeers();

	std::string m_clientVersion;
	NodeMode m_mode = NodeMode::Full;

	unsigned short m_listenPort;

	BlockChain const* m_chain = nullptr;
	ba::io_service m_ioService;
	bi::tcp::acceptor m_acceptor;
	bi::tcp::socket m_socket;

	UPnP* m_upnp = nullptr;
	bi::tcp::endpoint m_public;
	KeyPair m_key;

	unsigned m_networkId;
	std::map<Public, std::weak_ptr<PeerSession>> m_peers;

	std::vector<bytes> m_incomingTransactions;
	std::vector<bytes> m_incomingBlocks;
	std::vector<bytes> m_unknownParentBlocks;
	std::vector<Public> m_freePeers;
	std::map<Public, std::pair<bi::tcp::endpoint, unsigned>> m_incomingPeers;

	h256 m_latestBlockSent;
	std::set<h256> m_transactionsSent;

	std::chrono::steady_clock::time_point m_lastPeersRequest;
	unsigned m_idealPeerCount = 5;

	std::vector<bi::address_v4> m_addresses;
	std::vector<bi::address_v4> m_peerAddresses;

	bool m_accepting = false;
};

}
