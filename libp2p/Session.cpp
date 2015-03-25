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
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include "Session.h"

#include <chrono>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/StructuredLogger.h>
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

Session::Session(Host* _s, RLPXFrameIO* _io, std::shared_ptr<Peer> const& _n, PeerSessionInfo _info):
	m_server(_s),
	m_io(_io),
	m_socket(m_io->socket()),
	m_peer(_n),
	m_info(_info),
	m_ping(chrono::steady_clock::time_point::max())
{
	m_lastReceived = m_connect = chrono::steady_clock::now();
}

Session::~Session()
{
	clogS(NetMessageSummary) << "Closing Peer Session :-(";
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
	delete m_io;
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

bool Session::interpret(PacketType _t, RLP const& _r)
{
	m_lastReceived = chrono::steady_clock::now();

	clogS(NetRight) << _t << _r;
	try		// Generic try-catch block designed to capture RLP format errors - TODO: give decent diagnostics, make a bit more specific over what is caught.
	{

	switch (_t)
	{
	case DisconnectPacket:
	{
		string reason = "Unspecified";
		auto r = (DisconnectReason)_r[0].toInt<int>();
		if (!_r[0].isInt())
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
		for (unsigned i = 0; i < _r.itemCount(); ++i)
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
		for (auto const& i: m_capabilities)
			if (_t >= i.second->m_idOffset && _t - i.second->m_idOffset < i.second->hostCapability()->messageCount())
			{
				if (i.second->m_enabled)
					return i.second->interpret(_t - i.second->m_idOffset, _r);
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
	return _s.append((unsigned)_id).appendList(_args);
}

void Session::sealAndSend(RLPStream& _s)
{
	bytes b;
	_s.swapOut(b);
	send(move(b));
}

bool Session::checkPacket(bytesConstRef _msg)
{
	if (_msg.size() < 2)
		return false;
	if (_msg[0] > 0x7f)
		return false;
	RLP r(_msg.cropped(1));
	if (r.actualSize() + 1 != _msg.size())
		return false;
	return true;
}

void Session::send(bytes&& _msg)
{
	clogS(NetLeft) << RLP(bytesConstRef(&_msg).cropped(1));

	bytesConstRef msg(&_msg);
	if (!checkPacket(msg))
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
	m_io->writeSingleFramePacket(&bytes, m_writeQueue[0]);
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
	StructuredLogger::p2pDisconnected(
		m_info.id.abridged(),
		m_peer->peerEndpoint(),
		m_server->peerCount()
	);
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
	ping();
	doRead();
}

void Session::doRead()
{
	// ignore packets received while waiting to disconnect.
	if (m_dropped)
		return;

	auto self(shared_from_this());
	ba::async_read(m_socket, boost::asio::buffer(m_data, h256::size), [this,self](boost::system::error_code ec, std::size_t length)
	{
		if (ec && ec.category() != boost::asio::error::get_misc_category() && ec.value() != boost::asio::error::eof)
		{
			clogS(NetWarn) << "Error reading: " << ec.message();
			drop(TCPError);
		}
		else if (ec && length == 0)
			return;
		else
		{
			/// authenticate and decrypt header
			bytesRef header(m_data.data(), h256::size);
			if (!m_io->authAndDecryptHeader(header))
			{
				clog(NetWarn) << "header decrypt failed";
				drop(BadProtocol); // todo: better error
				return;
			}

			/// check frame size
			uint32_t frameSize = (m_data[0] * 256 + m_data[1]) * 256 + m_data[2];
			if (frameSize >= (uint32_t)1 << 24)
			{
				clog(NetWarn) << "frame size too large";
				drop(BadProtocol);
				return;
			}
			
			/// rlp of header has protocol-type, sequence-id[, total-packet-size]
			bytes headerRLP(13);
			bytesConstRef(m_data.data(), h128::size).cropped(3).copyTo(&headerRLP);
			
			/// read padded frame and mac
			auto tlen = frameSize + ((16 - (frameSize % 16)) % 16) + h128::size;
			ba::async_read(m_socket, boost::asio::buffer(m_data, tlen), [this, self, headerRLP, frameSize, tlen](boost::system::error_code ec, std::size_t length)
			{
				if (ec && ec.category() != boost::asio::error::get_misc_category() && ec.value() != boost::asio::error::eof)
				{
					clogS(NetWarn) << "Error reading: " << ec.message();
					drop(TCPError);
				}
				else if (ec && length == 0)
					return;
				else
				{
					if (!m_io->authAndDecryptFrame(bytesRef(m_data.data(), tlen)))
					{
						clog(NetWarn) << "frame decrypt failed";
						drop(BadProtocol); // todo: better error
						return;
					}
					
					bytesConstRef frame(m_data.data(), frameSize);
					if (!checkPacket(frame))
					{
						cerr << "Received " << frame.size() << ": " << toHex(frame) << endl;
						clogS(NetWarn) << "INVALID MESSAGE RECEIVED";
						disconnect(BadProtocol);
						return;
					}
					else
					{
						auto packetType = (PacketType)RLP(frame.cropped(0, 1)).toInt<unsigned>();
						RLP r(frame.cropped(1));
						if (!interpret(packetType, r))
							clogS(NetWarn) << "Couldn't interpret packet." << RLP(r);
					}
					doRead();
				}
			});
		}
	});
}
