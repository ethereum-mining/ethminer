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
/** @file Session.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Session.h"

#include <chrono>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libethcore/Exceptions.h>
#include "Host.h"
#include "Capability.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

#if defined(clogS)
#undef clogS
#endif
#define clogS(X) dev::LogOutputStream<X, true>(false) << "| " << std::setw(2) << m_socket.native_handle() << "] "

Session::Session(Host* _s, bi::tcp::socket _socket, std::shared_ptr<Peer> const& _n):
	m_server(_s),
	m_socket(std::move(_socket)),
	m_peer(_n),
	m_info({NodeId(), "?", m_socket.remote_endpoint().address().to_string(), 0, chrono::steady_clock::duration(0), CapDescSet(), 0, map<string, string>()}),
	m_ping(chrono::steady_clock::time_point::max())
{
	m_lastReceived = m_connect = chrono::steady_clock::now();
}

Session::~Session()
{
	m_peer->m_lastConnected = m_peer->m_lastAttempted - chrono::seconds(1);

	// Read-chain finished for one reason or another.
	for (auto& i: m_capabilities)
		i.second.reset();

	try
	{
		if (m_socket.is_open())
		{
			boost::system::error_code ec;
			m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
			m_socket.close();
		}
	}
	catch (...){}
}

NodeId Session::id() const
{
	return m_peer ? m_peer->id : NodeId();
}

void Session::addRating(unsigned _r)
{
	if (m_peer)
	{
		m_peer->m_rating += _r;
		m_peer->m_score += _r;
	}
}

int Session::rating() const
{
	return m_peer->m_rating;
}

template <class T> vector<T> randomSelection(vector<T> const& _t, unsigned _n)
{
	if (_t.size() <= _n)
		return _t;
	vector<T> ret = _t;
	while (ret.size() > _n)
	{
		auto i = ret.begin();
		advance(i, rand() % ret.size());
		ret.erase(i);
	}
	return ret;
}

// TODO: P2P integration: replace w/asio post -> serviceNodesRequest()
void Session::ensureNodesRequested()
{
	if (isConnected() && !m_weRequestedNodes)
	{
		m_weRequestedNodes = true;
		RLPStream s;
		sealAndSend(prep(s, GetPeersPacket));
	}
}

void Session::serviceNodesRequest()
{
	if (!m_theyRequestedNodes)
		return;

// TODO: P2P reimplement, as per TCP "close nodes" gossip specifications (WiP)
//	auto peers = m_server->potentialPeers(m_knownNodes);
	Peers peers;
	if (peers.empty())
	{
		addNote("peers", "requested");
		return;
	}

	// note this should cost them...
	RLPStream s;
	prep(s, PeersPacket, min<unsigned>(10, peers.size()));
	auto rs = randomSelection(peers, 10);
	for (auto const& i: rs)
	{
		clogS(NetTriviaDetail) << "Sending peer " << i.id.abridged() << i.peerEndpoint();
		if (i.peerEndpoint().address().is_v4())
			s.appendList(3) << bytesConstRef(i.peerEndpoint().address().to_v4().to_bytes().data(), 4) << i.peerEndpoint().port() << i.id;
		else// if (i.second.address().is_v6()) - assumed
			s.appendList(3) << bytesConstRef(i.peerEndpoint().address().to_v6().to_bytes().data(), 16) << i.peerEndpoint().port() << i.id;
	}
	sealAndSend(s);
	m_theyRequestedNodes = false;
	addNote("peers", "done");
}

bool Session::interpret(RLP const& _r)
{
	m_lastReceived = chrono::steady_clock::now();

	clogS(NetRight) << _r;
	try		// Generic try-catch block designed to capture RLP format errors - TODO: give decent diagnostics, make a bit more specific over what is caught.
	{

	switch ((PacketType)_r[0].toInt<unsigned>())
	{
	case HelloPacket:
	{
		// TODO: P2P first pass, implement signatures. if signature fails, drop connection. if egress, flag node's endpoint as stale.
		// Move auth to Host so we consolidate authentication logic and eschew peer deduplication logic.
		// Move all node-lifecycle information into Host.
		// Finalize peer-lifecycle properties vs node lifecycle.
		
		m_protocolVersion = _r[1].toInt<unsigned>();
		auto clientVersion = _r[2].toString();
		auto caps = _r[3].toVector<CapDesc>();
		auto listenPort = _r[4].toInt<unsigned short>();
		auto id = _r[5].toHash<NodeId>();

		// clang error (previously: ... << hex << caps ...)
		// "'operator<<' should be declared prior to the call site or in an associated namespace of one of its arguments"
		stringstream capslog;
		for (auto cap: caps)
			capslog << "(" << cap.first << "," << dec << cap.second << ")";

		clogS(NetMessageSummary) << "Hello: " << clientVersion << "V[" << m_protocolVersion << "]" << id.abridged() << showbase << capslog.str() << dec << listenPort;

		if (m_server->id() == id)
		{
			// Already connected.
			clogS(NetWarn) << "Connected to ourself under a false pretext. We were told this peer was id" << id.abridged();
			disconnect(LocalIdentity);
			return true;
		}

		// if peer and connection have id, check for UnexpectedIdentity
		if (!id)
		{
			disconnect(NullIdentity);
			return true;
		}
		else if (!m_peer->id)
		{
			m_peer->id = id;
			m_peer->endpoint.tcp.port(listenPort);
		}
		else if (m_peer->id != id)
		{
			// TODO p2p: FIXME. Host should catch this and reattempt adding node to table.
			m_peer->id = id;
			m_peer->m_score = 0;
			m_peer->m_rating = 0;
//			disconnect(UnexpectedIdentity);
//			return true;
		}

		if (m_server->havePeerSession(id))
		{
			// Already connected.
			clogS(NetWarn) << "Already connected to a peer with id" << id.abridged();
			// Possible that two nodes continually connect to each other with exact same timing.
			this_thread::sleep_for(chrono::milliseconds(rand() % 100));
			disconnect(DuplicatePeer);
			return true;
		}
		
		if (m_peer->isOffline())
			m_peer->m_lastConnected = chrono::system_clock::now();

		if (m_protocolVersion != m_server->protocolVersion())
		{
			disconnect(IncompatibleProtocol);
			return true;
		}
		
		m_info.clientVersion = clientVersion;
		m_info.host = m_socket.remote_endpoint().address().to_string();
		m_info.port = listenPort;
		m_info.lastPing = std::chrono::steady_clock::duration();
		m_info.caps = _r[3].toSet<CapDesc>();
		m_info.socket = (unsigned)m_socket.native_handle();
		m_info.notes = map<string, string>();

		m_server->registerPeer(shared_from_this(), caps);
		break;
	}
	case DisconnectPacket:
	{
		string reason = "Unspecified";
		auto r = (DisconnectReason)_r[1].toInt<int>();
		if (!_r[1].isInt())
			drop(BadProtocol);
		else
		{
			reason = reasonOf(r);
			clogS(NetMessageSummary) << "Disconnect (reason: " << reason << ")";
			drop(DisconnectRequested);
		}
		break;
	}
	case PingPacket:
	{
		clogS(NetTriviaSummary) << "Ping";
		RLPStream s;
		sealAndSend(prep(s, PongPacket));
		break;
	}
	case PongPacket:
		m_info.lastPing = std::chrono::steady_clock::now() - m_ping;
		clogS(NetTriviaSummary) << "Latency: " << chrono::duration_cast<chrono::milliseconds>(m_info.lastPing).count() << " ms";
		break;
	case GetPeersPacket:
	{
		// Disabled for interop testing.
		// GetPeers/PeersPacket will be modified to only exchange new nodes which it's peers are interested in.
		break;
		
		clogS(NetTriviaSummary) << "GetPeers";
		m_theyRequestedNodes = true;
		serviceNodesRequest();
		break;
	}
	case PeersPacket:
		// Disabled for interop testing.
		// GetPeers/PeersPacket will be modified to only exchange new nodes which it's peers are interested in.
		break;
			
		clogS(NetTriviaSummary) << "Peers (" << dec << (_r.itemCount() - 1) << " entries)";
		m_weRequestedNodes = false;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			bi::address peerAddress;
			if (_r[i][0].size() == 16)
				peerAddress = bi::address_v6(_r[i][0].toHash<FixedHash<16>>().asArray());
			else if (_r[i][0].size() == 4)
				peerAddress = bi::address_v4(_r[i][0].toHash<FixedHash<4>>().asArray());
			else
			{
				cwarn << "Received bad peer packet:" << _r;
				disconnect(BadProtocol);
				return true;
			}
			auto ep = bi::tcp::endpoint(peerAddress, _r[i][1].toInt<short>());
			NodeId id = _r[i][2].toHash<NodeId>();
			
			clogS(NetAllDetail) << "Checking: " << ep << "(" << id.abridged() << ")";
//			clogS(NetAllDetail) << "Checking: " << ep << "(" << id.abridged() << ")" << isPrivateAddress(peerAddress) << this->id().abridged() << isPrivateAddress(endpoint().address()) << m_server->m_peers.count(id) << (m_server->m_peers.count(id) ? isPrivateAddress(m_server->m_peers.at(id)->address.address()) : -1);

			// todo: draft spec: ignore if dist(us,item) - dist(us,them) > 1
			
			// TODO: isPrivate
			if (!m_server->m_netPrefs.localNetworking && isPrivateAddress(peerAddress))
				goto CONTINUE;	// Private address. Ignore.

			if (!id)
				goto LAMEPEER;	// Null identity. Ignore.

			if (m_server->id() == id)
				goto LAMEPEER;	// Just our info - we already have that.

			if (id == this->id())
				goto LAMEPEER;	// Just their info - we already have that.

			if (!ep.port())
				goto LAMEPEER;	// Zero port? Don't think so.

			if (ep.port() >= /*49152*/32768)
				goto LAMEPEER;	// Private port according to IANA.

			// OK passed all our checks. Assume it's good.
			addRating(1000);
			m_server->addNode(id, ep.address().to_string(), ep.port(), ep.port());
			clogS(NetTriviaDetail) << "New peer: " << ep << "(" << id .abridged()<< ")";
			CONTINUE:;
			LAMEPEER:;
		}
		break;
	default:
	{
		auto id = _r[0].toInt<unsigned>();
		for (auto const& i: m_capabilities)
			if (id >= i.second->m_idOffset && id - i.second->m_idOffset < i.second->hostCapability()->messageCount())
			{
				if (i.second->m_enabled)
					return i.second->interpret(id - i.second->m_idOffset, _r);
				else
					return true;
			}
		return false;
	}
	}
	}
	catch (std::exception const& _e)
	{
		clogS(NetWarn) << "Peer causing an exception:" << _e.what() << _r;
		disconnect(BadProtocol);
		return true;
	}
	return true;
}

void Session::ping()
{
	RLPStream s;
	sealAndSend(prep(s, PingPacket));
	m_ping = std::chrono::steady_clock::now();
}

RLPStream& Session::prep(RLPStream& _s, PacketType _id, unsigned _args)
{
	return prep(_s).appendList(_args + 1).append((unsigned)_id);
}

RLPStream& Session::prep(RLPStream& _s)
{
	return _s.appendRaw(bytes(8, 0));
}

void Session::sealAndSend(RLPStream& _s)
{
	bytes b;
	_s.swapOut(b);
	m_server->seal(b);
	send(move(b));
}

bool Session::checkPacket(bytesConstRef _msg)
{
	if (_msg.size() < 8)
		return false;
	if (!(_msg[0] == 0x22 && _msg[1] == 0x40 && _msg[2] == 0x08 && _msg[3] == 0x91))
		return false;
	uint32_t len = ((_msg[4] * 256 + _msg[5]) * 256 + _msg[6]) * 256 + _msg[7];
	if (_msg.size() != len + 8)
		return false;
	RLP r(_msg.cropped(8));
	if (r.actualSize() != len)
		return false;
	return true;
}

void Session::send(bytesConstRef _msg)
{
	send(_msg.toBytes());
}

void Session::send(bytes&& _msg)
{
	clogS(NetLeft) << RLP(bytesConstRef(&_msg).cropped(8));

	if (!checkPacket(bytesConstRef(&_msg)))
		clogS(NetWarn) << "INVALID PACKET CONSTRUCTED!";

	if (!m_socket.is_open())
		return;

	bool doWrite = false;
	{
		Guard l(x_writeQueue);
		m_writeQueue.push_back(_msg);
		doWrite = (m_writeQueue.size() == 1);
	}

	if (doWrite)
		write();
}

void Session::write()
{
	const bytes& bytes = m_writeQueue[0];
	auto self(shared_from_this());
	ba::async_write(m_socket, ba::buffer(bytes), [this, self](boost::system::error_code ec, std::size_t /*length*/)
	{
		// must check queue, as write callback can occur following dropped()
		if (ec)
		{
			clogS(NetWarn) << "Error sending: " << ec.message();
			drop(TCPError);
			return;
		}
		else
		{
			Guard l(x_writeQueue);
			m_writeQueue.pop_front();
			if (m_writeQueue.empty())
				return;
		}
		write();
	});
}

void Session::drop(DisconnectReason _reason)
{
	if (m_dropped)
		return;
	if (m_socket.is_open())
		try
		{
			clogS(NetConnect) << "Closing " << m_socket.remote_endpoint() << "(" << reasonOf(_reason) << ")";
			boost::system::error_code ec;
			m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
			m_socket.close();
		}
		catch (...) {}

	if (m_peer)
	{
		if (_reason != m_peer->m_lastDisconnect || _reason == NoDisconnect || _reason == ClientQuit || _reason == DisconnectRequested)
			m_peer->m_failedAttempts = 0;
		m_peer->m_lastDisconnect = _reason;
		if (_reason == BadProtocol)
		{
			m_peer->m_rating /= 2;
			m_peer->m_score /= 2;
		}
	}
	m_dropped = true;
}

void Session::disconnect(DisconnectReason _reason)
{
	clogS(NetConnect) << "Disconnecting (our reason:" << reasonOf(_reason) << ")";
	if (m_socket.is_open())
	{
		RLPStream s;
		prep(s, DisconnectPacket, 1) << (int)_reason;
		sealAndSend(s);
	}
	drop(_reason);
}

void Session::start()
{
	RLPStream s;
	prep(s, HelloPacket, 5)
					<< m_server->protocolVersion()
					<< m_server->m_clientVersion
					<< m_server->caps()
					<< m_server->m_tcpPublic.port()
					<< m_server->id();
	sealAndSend(s);
	ping();
	doRead();
}

void Session::doRead()
{
	// ignore packets received while waiting to disconnect
	if (m_dropped)
		return;
	
	auto self(shared_from_this());
	m_socket.async_read_some(boost::asio::buffer(m_data), [this,self](boost::system::error_code ec, std::size_t length)
	{
		// If error is end of file, ignore
		if (ec && ec.category() != boost::asio::error::get_misc_category() && ec.value() != boost::asio::error::eof)
		{
			// got here with length of 1241...
			clogS(NetWarn) << "Error reading: " << ec.message();
			drop(TCPError);
		}
		else if (ec && length == 0)
		{
			return;
		}
		else
		{
			try
			{
				m_incoming.resize(m_incoming.size() + length);
				memcpy(m_incoming.data() + m_incoming.size() - length, m_data.data(), length);
				while (m_incoming.size() > 8)
				{
					if (m_incoming[0] != 0x22 || m_incoming[1] != 0x40 || m_incoming[2] != 0x08 || m_incoming[3] != 0x91)
					{
						clogS(NetWarn) << "INVALID SYNCHRONISATION TOKEN; expected = 22400891; received = " << toHex(bytesConstRef(m_incoming.data(), 4));
						disconnect(BadProtocol);
						return;
					}
					else
					{
						uint32_t len = fromBigEndian<uint32_t>(bytesConstRef(m_incoming.data() + 4, 4));
						uint32_t tlen = len + 8;
						if (m_incoming.size() < tlen)
							break;

						// enough has come in.
						auto data = bytesConstRef(m_incoming.data(), tlen);
						if (!checkPacket(data))
						{
							cerr << "Received " << len << ": " << toHex(bytesConstRef(m_incoming.data() + 8, len)) << endl;
							clogS(NetWarn) << "INVALID MESSAGE RECEIVED";
							disconnect(BadProtocol);
							return;
						}
						else
						{
							RLP r(data.cropped(8));
							if (!interpret(r))
							{
								// error - bad protocol
								clogS(NetWarn) << "Couldn't interpret packet." << RLP(r);
								// Just wasting our bandwidth - perhaps reduce rating?
								//return;
							}
						}
						memmove(m_incoming.data(), m_incoming.data() + tlen, m_incoming.size() - tlen);
						m_incoming.resize(m_incoming.size() - tlen);
					}
				}
				doRead();
			}
			catch (Exception const& _e)
			{
				clogS(NetWarn) << "ERROR: " << diagnostic_information(_e);
				drop(BadProtocol);
			}
			catch (std::exception const& _e)
			{
				clogS(NetWarn) << "ERROR: " << _e.what();
				drop(BadProtocol);
			}
		}
	});
}
