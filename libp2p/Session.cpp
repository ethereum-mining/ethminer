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

Session::Session(Host* _h, RLPXFrameCoder* _io, std::shared_ptr<RLPXSocket> const& _s, std::shared_ptr<Peer> const& _n, PeerSessionInfo _info):
	m_server(_h),
	m_io(_io),
	m_socket(_s),
	m_peer(_n),
	m_info(_info),
	m_ping(chrono::steady_clock::time_point::max())
{
	m_peer->m_lastDisconnect = NoDisconnect;
	m_lastReceived = m_connect = chrono::steady_clock::now();
	DEV_GUARDED(x_info)
		m_info.socketId = m_socket->ref().native_handle();
}

Session::~Session()
{
	ThreadContext tc(info().id.abridged());
	ThreadContext tc2(info().clientVersion);
	clog(NetMessageSummary) << "Closing peer session :-(";
	m_peer->m_lastConnected = m_peer->m_lastAttempted - chrono::seconds(1);

	// Read-chain finished for one reason or another.
	for (auto& i: m_capabilities)
		i.second.reset();

	try
	{
		bi::tcp::socket& socket = m_socket->ref();
		if (socket.is_open())
		{
			boost::system::error_code ec;
			socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
			socket.close();
		}
	}
	catch (...){}
	delete m_io;
}

ReputationManager& Session::repMan() const
{
	return m_server->repMan();
}

NodeId Session::id() const
{
	return m_peer ? m_peer->id : NodeId();
}

void Session::addRating(int _r)
{
	if (m_peer)
	{
		m_peer->m_rating += _r;
		m_peer->m_score += _r;
		if (_r >= 0)
			m_peer->noteSessionGood();
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
	ThreadContext tc(info().id.abridged() + "/" + info().clientVersion);

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
		clog(NetTriviaDetail) << "Sending peer " << i.id << i.endpoint;
		if (i.endpoint.address.is_v4())
			s.appendList(3) << bytesConstRef(i.endpoint.address.to_v4().to_bytes().data(), 4) << i.endpoint.tcpPort << i.id;
		else// if (i.second.address().is_v6()) - assumed
			s.appendList(3) << bytesConstRef(i.endpoint.address.to_v6().to_bytes().data(), 16) << i.endpoint.tcpPort << i.id;
	}
	sealAndSend(s);
	m_theyRequestedNodes = false;
	addNote("peers", "done");
}

bool Session::readPacket(uint16_t _capId, PacketType _t, RLP const& _r)
{
	m_lastReceived = chrono::steady_clock::now();
	clog(NetRight) << _t << _r;
	try		// Generic try-catch block designed to capture RLP format errors - TODO: give decent diagnostics, make a bit more specific over what is caught.
	{
		// v4 frame headers are useless, offset packet type used
		// v5 protocol type is in header, packet type not offset
		if (_capId == 0 && _t < UserPacket)
			return interpret(_t, _r);
		if (m_info.protocolVersion >= 5)
			for (auto const& i: m_capabilities)
				if (_capId == (uint16_t)i.first.second)
					return i.second->m_enabled ? i.second->interpret(_t, _r) : true;
		if (m_info.protocolVersion <= 4)
			for (auto const& i: m_capabilities)
				if (_t >= (int)i.second->m_idOffset && _t - i.second->m_idOffset < i.second->hostCapability()->messageCount())
					return i.second->m_enabled ? i.second->interpret(_t - i.second->m_idOffset, _r) : true;
		return false;
	}
	catch (std::exception const& _e)
	{
		clog(NetWarn) << "Exception caught in p2p::Session::interpret(): " << _e.what() << ". PacketType: " << _t << ". RLP: " << _r;
		disconnect(BadProtocol);
		return true;
	}
	return true;
}

bool Session::interpret(PacketType _t, RLP const& _r)
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
			clog(NetMessageSummary) << "Disconnect (reason: " << reason << ")";
			drop(DisconnectRequested);
		}
		break;
	}
	case PingPacket:
	{
		clog(NetTriviaSummary) << "Ping";
		RLPStream s;
		sealAndSend(prep(s, PongPacket));
		break;
	}
	case PongPacket:
		DEV_GUARDED(x_info)
		{
			m_info.lastPing = std::chrono::steady_clock::now() - m_ping;
			clog(NetTriviaSummary) << "Latency: " << chrono::duration_cast<chrono::milliseconds>(m_info.lastPing).count() << " ms";
		}
		break;
	case GetPeersPacket:
	case PeersPacket:
		break;
	default:
		return false;
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
	if (_msg[0] > 0x7f || _msg.size() < 2)
		return false;
	if (RLP(_msg.cropped(1)).actualSize() + 1 != _msg.size())
		return false;
	return true;
}

void Session::send(bytes&& _msg)
{
	bytesConstRef msg(&_msg);
	clog(NetLeft) << RLP(msg.cropped(1));
	if (!checkPacket(msg))
		clog(NetWarn) << "INVALID PACKET CONSTRUCTED!";

	if (!m_socket->ref().is_open())
		return;

	bool doWrite = false;
	{
		Guard l(x_writeQueue);
		m_writeQueue.push_back(std::move(_msg));
		doWrite = (m_writeQueue.size() == 1);
	}

	if (doWrite)
		write();
}

void Session::write()
{
	bytes const* out;
	DEV_GUARDED(x_writeQueue)
	{
		m_io->writeSingleFramePacket(&m_writeQueue[0], m_writeQueue[0]);
		out = &m_writeQueue[0];
	}
	auto self(shared_from_this());
	ba::async_write(m_socket->ref(), ba::buffer(*out), [this, self](boost::system::error_code ec, std::size_t /*length*/)
	{
		ThreadContext tc(info().id.abridged());
		ThreadContext tc2(info().clientVersion);
		// must check queue, as write callback can occur following dropped()
		if (ec)
		{
			clog(NetWarn) << "Error sending: " << ec.message();
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
	bi::tcp::socket& socket = m_socket->ref();
	if (socket.is_open())
		try
		{
			clog(NetConnect) << "Closing " << socket.remote_endpoint() << "(" << reasonOf(_reason) << ")";
			boost::system::error_code ec;
			socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
			socket.close();
		}
		catch (...) {}

	m_peer->m_lastDisconnect = _reason;
	if (_reason == BadProtocol)
	{
		m_peer->m_rating /= 2;
		m_peer->m_score /= 2;
	}
	m_dropped = true;
}

void Session::disconnect(DisconnectReason _reason)
{
	clog(NetConnect) << "Disconnecting (our reason:" << reasonOf(_reason) << ")";
	size_t peerCount = m_server->peerCount(); //needs to  be outside of lock to avoid deadlocking with other thread that capture x_info/x_sessions in reverse order
	DEV_GUARDED(x_info)
		StructuredLogger::p2pDisconnected(
			m_info.id.abridged(),
			m_peer->endpoint, // TODO: may not be 100% accurate
			peerCount
		);
	if (m_socket->ref().is_open())
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
	ba::async_read(m_socket->ref(), boost::asio::buffer(m_data, h256::size), [this,self](boost::system::error_code ec, std::size_t length)
	{
		ThreadContext tc(info().id.abridged());
		ThreadContext tc2(info().clientVersion);
		if (!checkRead(h256::size, ec, length))
			return;
		else if (!m_io->authAndDecryptHeader(bytesRef(m_data.data(), length)))
		{
			clog(NetWarn) << "header decrypt failed";
			drop(BadProtocol); // todo: better error
			return;
		}

		RLPXFrameInfo header;
		try
		{
			header = RLPXFrameInfo(bytesConstRef(m_data.data(), length));
		}
		catch (std::exception const& _e)
		{
			clog(NetWarn) << "Exception decoding frame header RLP:" << _e.what() << bytesConstRef(m_data.data(), h128::size).cropped(3);
			drop(BadProtocol);
			return;
		}

		/// read padded frame and mac
		auto tlen = header.length + header.padding + h128::size;
		ba::async_read(m_socket->ref(), boost::asio::buffer(m_data, tlen), [this, self, header, tlen](boost::system::error_code ec, std::size_t length)
		{
			ThreadContext tc(info().id.abridged());
			ThreadContext tc2(info().clientVersion);
			if (!checkRead(tlen, ec, length))
				return;
			else if (!m_io->authAndDecryptFrame(bytesRef(m_data.data(), tlen)))
			{
				clog(NetWarn) << "frame decrypt failed";
				drop(BadProtocol); // todo: better error
				return;
			}

			bytesConstRef frame(m_data.data(), header.length);
			if (!checkPacket(frame))
			{
				cerr << "Received " << frame.size() << ": " << toHex(frame) << endl;
				clog(NetWarn) << "INVALID MESSAGE RECEIVED";
				disconnect(BadProtocol);
				return;
			}
			else
			{
				auto packetType = (PacketType)RLP(frame.cropped(0, 1)).toInt<unsigned>();
				RLP r(frame.cropped(1));
				if (!readPacket(header.protocolId, packetType, r))
					clog(NetWarn) << "Couldn't interpret packet." << RLP(r);
			}
			doRead();
		});
	});
}

bool Session::checkRead(std::size_t _expected, boost::system::error_code _ec, std::size_t _length)
{
	if (_ec && _ec.category() != boost::asio::error::get_misc_category() && _ec.value() != boost::asio::error::eof)
	{
		clog(NetConnect) << "Error reading: " << _ec.message();
		drop(TCPError);
		return false;
	}
	else if (_ec && _length < _expected)
	{
		clog(NetWarn) << "Error reading - Abrupt peer disconnect: " << _ec.message();
		repMan().noteRude(*this);
		drop(TCPError);
		return false;
	}
	// If this fails then there's an unhandled asio error
	assert(_expected == _length);
	return true;
}
