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
	GetChain
};

class PeerSession: public std::enable_shared_from_this<PeerSession>
{
public:
	PeerSession(bi::tcp::socket _socket, uint _rNId);
	~PeerSession();

	void start();
	void disconnect();

	void ping();

private:
	void doRead();
	void doWrite(std::size_t length);
	bool interpret(RLP const& _r);

	RLPStream& prep(RLPStream& _s);
	void sealAndSend(RLPStream& _s);
	void send(bytes& _msg);

	bi::tcp::socket m_socket;
	std::array<byte, 65536> m_data;

	bytes m_incoming;
	std::string m_clientVersion;
	uint m_protocolVersion;
	uint m_networkId;
	uint m_reqNetworkId;

	std::vector<bytes> m_incomingTransactions;
	std::vector<bytes> m_incomingBlocks;
	std::vector<bi::tcp::endpoint> m_incomingPeers;

	std::chrono::steady_clock::time_point m_ping;
};

class PeerServer
{
public:
	/// Start server, listening for connections on the given port.
	PeerServer(uint _networkId, short _port);
	/// Start server, but don't listen.
	PeerServer(uint _networkId);

	/// Conduct I/O, polling, syncing, whatever.
	/// Ideally all time-consuming I/O is done in a background thread, but you get this call every 100ms or so anyway.
	void process();

	/// Connect to a peer explicitly.
	bool connect(std::string const& _addr = "127.0.0.1", uint _port = 30303);

	/// Sync with the BlockChain. It might contain one of our mined blocks, we might have new candidates from the network.
	void sync(BlockChain& _bc, TransactionQueue const&);

	/// Get an incoming transaction from the queue. @returns bytes() if nothing waiting.
	bytes const& incomingTransaction() { return NullBytes; }

	/// Remove incoming transaction from the queue. Make sure you've finished with the data from any previous incomingTransaction() calls.
	void popIncomingTransaction() {}

	void pingAll();

private:
	void doAccept();

	ba::io_service m_ioService;
	bi::tcp::acceptor m_acceptor;
	bi::tcp::socket m_socket;

	uint m_requiredNetworkId;
	std::vector<std::weak_ptr<PeerSession>> m_peers;
};


}
