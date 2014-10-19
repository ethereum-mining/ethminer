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

Session::Session(Host* _s, bi::tcp::socket _socket, bi::tcp::endpoint const& _manual):
	m_server(_s),
	m_socket(std::move(_socket)),
	m_node(nullptr),
	m_manualEndpoint(_manual)	// NOTE: the port on this shouldn't be used if it's zero.
{
	m_lastReceived = m_connect = std::chrono::steady_clock::now();

	m_info = PeerInfo({NodeId(), "?", m_manualEndpoint.address().to_string(), 0, std::chrono::steady_clock::duration(0), CapDescSet(), 0, map<string, string>()});
}

Session::Session(Host* _s, bi::tcp::socket _socket, std::shared_ptr<Node> const& _n, bool _force):
	m_server(_s),
	m_socket(std::move(_socket)),
	m_node(_n),
	m_manualEndpoint(_n->address),
	m_force(_force)
{
	m_lastReceived = m_connect = std::chrono::steady_clock::now();
	m_info = PeerInfo({m_node->id, "?", _n->address.address().to_string(), _n->address.port(), std::chrono::steady_clock::duration(0), CapDescSet(), 0, map<string, string>()});
}

Session::~Session()
{
	if (m_node)
	{
		if (id() && !isPermanentProblem(m_node->lastDisconnect) && !m_node->dead)
			m_server->m_ready += m_node->index;
		else
			m_node->lastConnected = m_node->lastAttempted - chrono::seconds(1);
	}

	// Read-chain finished for one reason or another.
	for (auto& i: m_capabilities)
		i.second.reset();

	try
	{
		if (m_socket.is_open())
			m_socket.close();
	}
	catch (...){}
}

NodeId Session::id() const
{
	return m_node ? m_node->id : NodeId();
}

void Session::addRating(unsigned _r)
{
	if (m_node)
	{
		m_node->rating += _r;
		m_node->score += _r;
	}
}

int Session::rating() const
{
	return m_node->rating;
}

bi::tcp::endpoint Session::endpoint() const
{
	if (m_socket.is_open() && m_node)
		try
		{
			return bi::tcp::endpoint(m_socket.remote_endpoint().address(), m_node->address.port());
		}
		catch (...) {}

	if (m_node)
		return m_node->address;

	return m_manualEndpoint;
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

void Session::ensureNodesRequested()
{
	if (isOpen() && !m_weRequestedNodes)
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

	auto peers = m_server->potentialPeers(m_knownNodes);
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
		clogS(NetTriviaDetail) << "Sending peer " << i.id.abridged() << i.address;
		if (i.address.address().is_v4())
			s.appendList(3) << bytesConstRef(i.address.address().to_v4().to_bytes().data(), 4) << i.address.port() << i.id;
		else// if (i.second.address().is_v6()) - assumed
			s.appendList(3) << bytesConstRef(i.address.address().to_v6().to_bytes().data(), 16) << i.address.port() << i.id;
		m_knownNodes.extendAll(i.index);
		m_knownNodes.unionWith(i.index);
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
		m_protocolVersion = _r[1].toInt<unsigned>();
		auto clientVersion = _r[2].toString();
		auto caps = _r[3].toVector<CapDesc>();
		auto listenPort = _r[4].toInt<unsigned short>();
		auto id = _r[5].toHash<NodeId>();

		// clang error (previously: ... << hex << caps ...)
		// "'operator<<' should be declared prior to the call site or in an associated namespace of one of its arguments"
		stringstream capslog;
		for (auto cap: caps)
			capslog << "(" << hex << cap.first << "," << hex << cap.second << ")";

		clogS(NetMessageSummary) << "Hello: " << clientVersion << "V[" << m_protocolVersion << "]" << id.abridged() << showbase << capslog.str() << dec << listenPort;

		if (m_server->id() == id)
		{
			// Already connected.
			clogS(NetWarn) << "Connected to ourself under a false pretext. We were told this peer was id" << m_info.id.abridged();
			disconnect(LocalIdentity);
			return true;
		}

		if (m_node && m_node->id != id)
		{
			if (m_force || m_node->idOrigin <= Origin::SelfThird)
				// SECURITY: We're forcing through the new ID, despite having been told
				clogS(NetWarn) << "Connected to node, but their ID has changed since last time. This could indicate a MitM attack. Allowing anyway...";
			else
			{
				clogS(NetWarn) << "Connected to node, but their ID has changed since last time. This could indicate a MitM attack. Disconnecting.";
				disconnect(UnexpectedIdentity);
				return true;
			}

			if (m_server->havePeer(id))
			{
				m_node->dead = true;
				disconnect(DuplicatePeer);
				return true;
			}
		}

		if (m_server->havePeer(id))
		{
			// Already connected.
			clogS(NetWarn) << "Already connected to a peer with id" << id.abridged();
			disconnect(DuplicatePeer);
			return true;
		}

		if (!id)
		{
			disconnect(NullIdentity);
			return true;
		}

		m_node = m_server->noteNode(id, bi::tcp::endpoint(m_socket.remote_endpoint().address(), listenPort), Origin::Self, false, !m_node || m_node->id == id ? NodeId() : m_node->id);
		if (m_node->isOffline())
			m_node->lastConnected = chrono::system_clock::now();
		m_knownNodes.extendAll(m_node->index);
		m_knownNodes.unionWith(m_node->index);

		if (m_protocolVersion != m_server->protocolVersion())
		{
			disconnect(IncompatibleProtocol);
			return true;
		}
		m_info = PeerInfo({id, clientVersion, m_socket.remote_endpoint().address().to_string(), listenPort, std::chrono::steady_clock::duration(), _r[3].toSet<CapDesc>(), (unsigned)m_socket.native_handle(), map<string, string>() });

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
        clogS(NetTriviaSummary) << "GetPeers";
		m_theyRequestedNodes = true;
		serviceNodesRequest();
		break;
	}
	case PeersPacket:
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
			clogS(NetAllDetail) << "Checking: " << ep << "(" << id.abridged() << ")" << isPrivateAddress(peerAddress) << this->id().abridged() << isPrivateAddress(endpoint().address()) << m_server->m_nodes.count(id) << (m_server->m_nodes.count(id) ? isPrivateAddress(m_server->m_nodes.at(id)->address.address()) : -1);

			if (isPrivateAddress(peerAddress) && !m_server->m_netPrefs.localNetworking)
				goto CONTINUE;	// Private address. Ignore.

			if (!id)
				goto CONTINUE;	// Null identity. Ignore.

			if (m_server->id() == id)
				goto CONTINUE;	// Just our info - we already have that.

			if (id == this->id())
				goto CONTINUE;	// Just their info - we already have that.

			// check that it's not us or one we already know:
			if (m_server->m_nodes.count(id))
			{
				/*	MEH. Far from an ideal solution. Leave alone for now.
				// Already got this node.
				// See if it's any better that ours or not...
				// This could be the public address of a known node.
				// SECURITY: remove this in beta - it's only for lazy connections and presents an easy attack vector.
				if (m_server->m_nodes.count(id) && isPrivateAddress(m_server->m_nodes.at(id)->address.address()) && ep.port() != 0)
					// Update address if the node if we now have a public IP for it.
					m_server->m_nodes[id]->address = ep;
				*/
				goto CONTINUE;
			}

			if (!ep.port())
				goto CONTINUE;	// Zero port? Don't think so.

			if (ep.port() >= 49152)
				goto CONTINUE;	// Private port according to IANA.

			// TODO: PoC-7:
			// Technically fine, but ignore for now to avoid peers passing on incoming ports until we can be sure that doesn't happen any more.
//			if (ep.port() < 30300 || ep.port() > 30305)
//				goto CONTINUE;	// Wierd port.

			// Avoid our random other addresses that they might end up giving us.
			for (auto i: m_server->m_addresses)
				if (ep.address() == i && ep.port() == m_server->listenPort())
					goto CONTINUE;

			// Check that we don't already know about this addr:port combination. If we are, assume the original is best.
			// SECURITY: Not a valid assumption in general. Should compare ID origins and pick the best or note uncertainty and weight each equally.
			for (auto const& i: m_server->m_nodes)
				if (i.second->address == ep)
					goto CONTINUE;		// Same address but a different node.

			// OK passed all our checks. Assume it's good.
			addRating(1000);
			m_server->noteNode(id, ep, m_node->idOrigin == Origin::Perfect ? Origin::PerfectThird : Origin::SelfThird, true);
			clogS(NetTriviaDetail) << "New peer: " << ep << "(" << id .abridged()<< ")";
			CONTINUE:;
		}
		break;
	default:
	{
		auto id = _r[0].toInt<unsigned>();
		for (auto const& i: m_capabilities)
			if (i.second->m_enabled && id >= i.second->m_idOffset && id - i.second->m_idOffset < i.second->hostCapability()->messageCount() && i.second->interpret(id - i.second->m_idOffset, _r))
				return true;
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

//	cerr << (void*)this << " writeImpl" << endl;
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
	cerr << (void*)this << " dropped" << endl;
	if (m_socket.is_open())
		try
		{
			clogS(NetConnect) << "Closing " << m_socket.remote_endpoint() << "(" << reasonOf(_reason) << ")";
			m_socket.close();
		}
		catch (...) {}

	if (m_node)
	{
		if (_reason != m_node->lastDisconnect || _reason == NoDisconnect || _reason == ClientQuit || _reason == DisconnectRequested)
			m_node->failedAttempts = 0;
		m_node->lastDisconnect = _reason;
		if (_reason == BadProtocol)
		{
			m_node->rating /= 2;
			m_node->score /= 2;
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
					<< m_server->m_public.port()
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
