/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 2 of the License, or
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

#include <memory>
#include <utility>
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <thread>
#include "RLP.h"
#include "Common.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace eth
{

class BlockChain;
class TransactionQueue;

enum PacketType
{
	Hello = 0,
	Disconnect,
	Ping,
	Pong,
	GetPeers = 0x10,
	Peers,
	Transactions,
	Blocks,
	GetChain,
	NotInChain,
	GetTransactions
};

class PeerServer;

struct PeerInfo
{
	std::string clientVersion;
	std::string host;
	short port;
	std::chrono::steady_clock::duration lastPing;
};

class PeerSession: public std::enable_shared_from_this<PeerSession>
{
	friend class PeerServer;

public:
	PeerSession(PeerServer* _server, bi::tcp::socket _socket, uint _rNId);
	~PeerSession();

	void start();
	void disconnect();

	void ping();

	bi::tcp::endpoint endpoint() const;	///< for other peers to connect to.

private:
	void dropped();
	void doRead();
	void doWrite(std::size_t length);
	bool interpret(RLP const& _r);

	static RLPStream& prep(RLPStream& _s);
	void sealAndSend(RLPStream& _s);
	void sendDestroy(bytes& _msg);
	void send(bytesConstRef _msg);

	PeerServer* m_server;
	bi::tcp::socket m_socket;
	std::array<byte, 65536> m_data;
	PeerInfo m_info;

	bytes m_incoming;
	uint m_protocolVersion;
	uint m_networkId;
	uint m_reqNetworkId;
	short m_listenPort;			///< Port that the remote client is listening on for connections. Useful for giving to peers.
	uint m_caps;

	std::chrono::steady_clock::time_point m_ping;
	std::chrono::steady_clock::time_point m_connect;
	std::chrono::steady_clock::time_point m_disconnect;

	unsigned m_rating;
	bool m_requireTransactions;

	std::set<h256> m_knownBlocks;
	std::set<h256> m_knownTransactions;
};

enum class NodeMode
{
	Full,
	PeerServer
};

struct UPnP;

class PeerServer
{
	friend class PeerSession;

public:
	/// Start server, listening for connections on the given port.
	PeerServer(std::string const& _clientVersion, BlockChain const& _ch, uint _networkId, short _port, NodeMode _m = NodeMode::Full, std::string const& _publicAddress = std::string());
	/// Start server, but don't listen.
	PeerServer(std::string const& _clientVersion, uint _networkId);
	~PeerServer();

	/// Connect to a peer explicitly.
	void connect(std::string const& _addr, uint _port = 30303) { connect(bi::tcp::endpoint(bi::address::from_string(_addr), _port)); }
	void connect(bi::tcp::endpoint const& _ep);

	/// Sync with the BlockChain. It might contain one of our mined blocks, we might have new candidates from the network.
	/// Conduct I/O, polling, syncing, whatever.
	/// Ideally all time-consuming I/O is done in a background thread or otherwise asynchronously, but you get this call every 100ms or so anyway.
	bool process(BlockChain& _bc, TransactionQueue&, Overlay& _o);
	bool process(BlockChain& _bc);

	/// Set ideal number of peers.
	void setIdealPeerCount(unsigned _n) { m_idealPeerCount = _n; }

	void setMode(NodeMode _m) { m_mode = _m; }
	void setVerbosity(unsigned _i) { m_verbosity = _i; }

	/// Get peer information.
	std::vector<PeerInfo> peers() const;

	/// Get number of peers connected; equivalent to, but faster than, peers().size().
	unsigned peerCount() const { return m_peers.size(); }

	/// Ping the peers, to update the latency information.
	void pingAll();

	/// Get the port we're listening on currently.
	short listenPort() const { return m_public.port(); }

private:
	void seal(bytes& _b);
	void populateAddresses();
	void determinePublic(std::string const& _publicAddress);
	void ensureAccepting();
	std::vector<bi::tcp::endpoint> potentialPeers();

	std::string m_clientVersion;
	NodeMode m_mode = NodeMode::Full;

	/**
	 * 0: Quiet - just errors on stderr.
	 * 1: Accepting/connecting/connected & one-off info.
	 * 2: Messages summary.
	 * 3: Messages detail.
	 * 6: Debug details.
	 * 8: Received raw.
	 * 9: Sent raw.
	 */
	unsigned m_verbosity = 4;

	short m_listenPort;

	BlockChain const* m_chain = nullptr;
	ba::io_service m_ioService;
	bi::tcp::acceptor m_acceptor;
	bi::tcp::socket m_socket;

	UPnP* m_upnp = nullptr;
	bi::tcp::endpoint m_public;

	uint m_requiredNetworkId;
	std::vector<std::weak_ptr<PeerSession>> m_peers;

	std::vector<bytes> m_incomingTransactions;
	std::vector<bytes> m_incomingBlocks;
	std::vector<bi::tcp::endpoint> m_incomingPeers;

	h256 m_latestBlockSent;
	std::set<h256> m_transactionsSent;

	std::chrono::steady_clock::time_point m_lastPeersRequest;
	unsigned m_idealPeerCount = 5;

	std::chrono::steady_clock::time_point m_lastFullProcess;

	std::vector<bi::address_v4> m_addresses;
	std::vector<bi::address_v4> m_peerAddresses;

	bool m_accepting = false;
};


}
