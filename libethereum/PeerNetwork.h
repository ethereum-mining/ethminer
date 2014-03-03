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
/** @file PeerNetwork.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <map>
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

bool isPrivateAddress(bi::address _addressToCheck);

class BlockChain;
class TransactionQueue;

struct NetWarn: public LogChannel { static const char* name() { return "!N!"; } static const int verbosity = 0; };
struct NetNote: public LogChannel { static const char* name() { return "*N*"; } static const int verbosity = 1; };
struct NetMessageSummary: public LogChannel { static const char* name() { return "-N-"; } static const int verbosity = 2; };
struct NetMessageDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 3; };
struct NetAllDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 6; };
struct NetRight: public LogChannel { static const char* name() { return ">N>"; } static const int verbosity = 8; };
struct NetLeft: public LogChannel { static const char* name() { return "<N<"; } static const int verbosity = 9; };

enum PacketType
{
	HelloPacket = 0,
	DisconnectPacket,
	PingPacket,
	PongPacket,
	GetPeersPacket = 0x10,
	PeersPacket,
	TransactionsPacket,
	BlocksPacket,
	GetChainPacket,
	NotInChainPacket,
	GetTransactionsPacket
};

enum DisconnectReason
{
	DisconnectRequested = 0,
	TCPError,
	BadProtocol,
	UselessPeer,
	TooManyPeers,
	DuplicatePeer,
	WrongGenesis,
	IncompatibleProtocol,
	ClientQuit
};

class PeerServer;

struct PeerInfo
{
	std::string clientVersion;
	std::string host;
	unsigned short port;
	std::chrono::steady_clock::duration lastPing;
};

class PeerSession: public std::enable_shared_from_this<PeerSession>
{
	friend class PeerServer;

public:
	PeerSession(PeerServer* _server, bi::tcp::socket _socket, uint _rNId, bi::address _peerAddress, unsigned short _peerPort = 0);
	~PeerSession();

	void start();
	void disconnect(int _reason);

	void ping();

	bool isOpen() const { return m_socket.is_open(); }

	static int protocolVersion();
	static int networkId();

	bi::tcp::endpoint endpoint() const;	///< for other peers to connect to.

private:
	void dropped();
	void doRead();
	void doWrite(std::size_t length);
	bool interpret(RLP const& _r);

	/// @returns true iff the _msg forms a valid message for sending or receiving on the network.
	static bool checkPacket(bytesConstRef _msg);

	static RLPStream& prep(RLPStream& _s);
	void sealAndSend(RLPStream& _s);
	void sendDestroy(bytes& _msg);
	void send(bytesConstRef _msg);
	PeerServer* m_server;

	bi::tcp::socket m_socket;
	std::array<byte, 65536> m_data;
	PeerInfo m_info;
	Public m_id;

	bytes m_incoming;
	uint m_protocolVersion;
	uint m_networkId;
	uint m_reqNetworkId;
	unsigned short m_listenPort;			///< Port that the remote client is listening on for connections. Useful for giving to peers.
	uint m_caps;

	std::chrono::steady_clock::time_point m_ping;
	std::chrono::steady_clock::time_point m_connect;
	std::chrono::steady_clock::time_point m_disconnect;

	uint m_rating;
	bool m_requireTransactions;

	std::set<h256> m_knownBlocks;
	std::set<h256> m_knownTransactions;
};

enum class NodeMode
{
	Full,
	PeerServer
};

class UPnP;

class PeerServer
{
	friend class PeerSession;

public:
	/// Start server, listening for connections on the given port.
	PeerServer(std::string const& _clientVersion, BlockChain const& _ch, uint _networkId, unsigned short _port, NodeMode _m = NodeMode::Full, std::string const& _publicAddress = std::string(), bool _upnp = true);
	/// Start server, but don't listen.
	PeerServer(std::string const& _clientVersion, uint _networkId, NodeMode _m = NodeMode::Full);
	~PeerServer();

	/// Connect to a peer explicitly.
	void connect(std::string const& _addr, unsigned short _port = 30303) noexcept;
	void connect(bi::tcp::endpoint const& _ep);

	/// Sync with the BlockChain. It might contain one of our mined blocks, we might have new candidates from the network.
	bool sync(BlockChain& _bc, TransactionQueue&, Overlay& _o);
	bool sync();

	/// Conduct I/O, polling, syncing, whatever.
	/// Ideally all time-consuming I/O is done in a background thread or otherwise asynchronously, but you get this call every 100ms or so anyway.
	/// This won't touch alter the blockchain.
	void process() { if (isInitialised()) m_ioService.poll(); }

	/// Set ideal number of peers.
	void setIdealPeerCount(unsigned _n) { m_idealPeerCount = _n; }

	void setMode(NodeMode _m) { m_mode = _m; }

	/// Get peer information.
	std::vector<PeerInfo> peers() const;

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

	uint m_requiredNetworkId;
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
