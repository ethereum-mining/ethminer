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
#include <libethcore/Exceptions.h>
#include "Host.h"
#include "Capability.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

#define clogS(X) dev::LogOutputStream<X, true>(false) << "| " << std::setw(2) << m_socket.native_handle() << "] "

Session::Session(Host* _s, bi::tcp::socket _socket, bi::address _peerAddress, unsigned short _peerPort):
	m_server(_s),
	m_socket(std::move(_socket)),
	m_listenPort(_peerPort),
	m_rating(0)
{
	m_disconnect = std::chrono::steady_clock::time_point::max();
	m_connect = std::chrono::steady_clock::now();
	m_info = PeerInfo({"?", _peerAddress.to_string(), m_listenPort, std::chrono::steady_clock::duration(0), set<string>(), 0});
}

Session::~Session()
{
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

bi::tcp::endpoint Session::endpoint() const
{
	if (m_socket.is_open())
		try
		{
			return bi::tcp::endpoint(m_socket.remote_endpoint().address(), m_listenPort);
		}
		catch (...){}

	return bi::tcp::endpoint();
}

bool Session::interpret(RLP const& _r)
{
	clogS(NetRight) << _r;
	switch (_r[0].toInt<unsigned>())
	{
	case HelloPacket:
	{
		m_protocolVersion = _r[1].toInt<unsigned>();
		auto clientVersion = _r[2].toString();
		auto caps = _r[3].toVector<string>();
		m_listenPort = _r[4].toInt<unsigned short>();
		m_id = _r[5].toHash<h512>();

		clogS(NetMessageSummary) << "Hello: " << clientVersion << "V[" << m_protocolVersion << "]" << m_id.abridged() << showbase << hex << caps << dec << m_listenPort;

		if (m_server->havePeer(m_id))
		{
			// Already connected.
			cwarn << "Already have peer id" << m_id.abridged();// << "at" << l->endpoint() << "rather than" << endpoint();
			disconnect(DuplicatePeer);
			return false;
		}
		if (!m_id)
		{
			disconnect(InvalidIdentity);
			return false;
		}
		if (m_protocolVersion != m_server->protocolVersion())
		{
			disconnect(IncompatibleProtocol);
			return false;
		}
		try
			{ m_info = PeerInfo({clientVersion, m_socket.remote_endpoint().address().to_string(), m_listenPort, std::chrono::steady_clock::duration(), _r[3].toSet<string>(), (unsigned)m_socket.native_handle()}); }
		catch (...)
		{
			disconnect(BadProtocol);
			return false;
		}

		m_server->registerPeer(shared_from_this(), caps);
		break;
	}
	case DisconnectPacket:
	{
		string reason = "Unspecified";
		if (_r[1].isInt())
			reason = reasonOf((DisconnectReason)_r[1].toInt<int>());

		clogS(NetMessageSummary) << "Disconnect (reason: " << reason << ")";
		if (m_socket.is_open())
			clogS(NetNote) << "Closing " << m_socket.remote_endpoint();
		else
			clogS(NetNote) << "Remote closed.";
		m_socket.close();
		return false;
	}
	case PingPacket:
	{
        clogS(NetTriviaSummary) << "Ping";
		RLPStream s;
		sealAndSend(prep(s).appendList(1) << PongPacket);
		break;
	}
	case PongPacket:
		m_info.lastPing = std::chrono::steady_clock::now() - m_ping;
        clogS(NetTriviaSummary) << "Latency: " << chrono::duration_cast<chrono::milliseconds>(m_info.lastPing).count() << " ms";
		break;
	case GetPeersPacket:
	{
        clogS(NetTriviaSummary) << "GetPeers";
		auto peers = m_server->potentialPeers();
		RLPStream s;
		prep(s).appendList(peers.size() + 1);
		s << PeersPacket;
		for (auto i: peers)
		{
			clogS(NetTriviaDetail) << "Sending peer " << i.first.abridged() << i.second;
			s.appendList(3) << bytesConstRef(i.second.address().to_v4().to_bytes().data(), 4) << i.second.port() << i.first;
		}
		sealAndSend(s);
		break;
	}
	case PeersPacket:
        clogS(NetTriviaSummary) << "Peers (" << dec << (_r.itemCount() - 1) << " entries)";
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			bi::address_v4 peerAddress(_r[i][0].toHash<FixedHash<4>>().asArray());
			auto ep = bi::tcp::endpoint(peerAddress, _r[i][1].toInt<short>());
			h512 id = _r[i][2].toHash<h512>();
			clogS(NetAllDetail) << "Checking: " << ep << "(" << id.abridged() << ")" << isPrivateAddress(peerAddress) << m_id.abridged() << isPrivateAddress(endpoint().address()) << m_server->m_incomingPeers.count(id) << (m_server->m_incomingPeers.count(id) ? isPrivateAddress(m_server->m_incomingPeers.at(id).first.address()) : -1);

			if (isPrivateAddress(peerAddress) && !m_server->m_netPrefs.localNetworking)
				goto CONTINUE;

			// check that it's not us or one we already know:
			if (!(m_id == id && isPrivateAddress(endpoint().address()) && (!m_server->m_incomingPeers.count(id) || isPrivateAddress(m_server->m_incomingPeers.at(id).first.address()))) && (!id || m_server->m_id == id || m_server->m_incomingPeers.count(id)))
				goto CONTINUE;

			// check that we're not already connected to addr:
			if (!ep.port())
				goto CONTINUE;
			for (auto i: m_server->m_addresses)
				if (ep.address() == i && ep.port() == m_server->listenPort())
					goto CONTINUE;
			for (auto i: m_server->m_incomingPeers)
				if (i.second.first == ep)
					goto CONTINUE;
			m_server->m_incomingPeers[id] = make_pair(ep, 0);
			m_server->m_freePeers.push_back(id);
			m_server->noteNewPeers();
			clogS(NetTriviaDetail) << "New peer: " << ep << "(" << id .abridged()<< ")";
			CONTINUE:;
		}
		break;
	default:
		for (auto const& i: m_capabilities)
			if (i.second->m_enabled && i.second->interpret(_r))
				return true;
		return false;
	}
	return true;
}

void Session::ping()
{
	RLPStream s;
	sealAndSend(prep(s).appendList(1) << PingPacket);
	m_ping = std::chrono::steady_clock::now();
}

void Session::getPeers()
{
	RLPStream s;
	sealAndSend(prep(s).appendList(1) << GetPeersPacket);
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
	sendDestroy(b);
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

void Session::sendDestroy(bytes& _msg)
{
	clogS(NetLeft) << RLP(bytesConstRef(&_msg).cropped(8));

	if (!checkPacket(bytesConstRef(&_msg)))
	{
		cwarn << "INVALID PACKET CONSTRUCTED!";
	}

	bytes buffer = bytes(std::move(_msg));
	writeImpl(buffer);
}

void Session::send(bytesConstRef _msg)
{
	clogS(NetLeft) << RLP(_msg.cropped(8));
	
	if (!checkPacket(_msg))
	{
		cwarn << "INVALID PACKET CONSTRUCTED!";
	}

	bytes buffer = bytes(_msg.toBytes());
	writeImpl(buffer);
}

void Session::writeImpl(bytes& _buffer)
{
//	cerr << (void*)this << " writeImpl" << endl;
	if (!m_socket.is_open())
		return;

	bool doWrite = false;
	{
		lock_guard<mutex> l(m_writeLock);
		m_writeQueue.push_back(_buffer);
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
//		cerr << (void*)this << " write.callback" << endl;

		// must check queue, as write callback can occur following dropped()
		if (ec)
		{
			cwarn << "Error sending: " << ec.message();
			dropped();
			return;
		}
		else
		{
			lock_guard<mutex> l(m_writeLock);
			m_writeQueue.pop_front();
			if (m_writeQueue.empty())
				return;
		}
		write();
	});
}

void Session::dropped()
{
//	cerr << (void*)this << " dropped" << endl;
	if (m_socket.is_open())
		try
		{
			clogS(NetConnect) << "Closing " << m_socket.remote_endpoint();
			m_socket.close();
		}
		catch (...) {}
}

void Session::disconnect(int _reason)
{
	clogS(NetConnect) << "Disconnecting (reason:" << reasonOf((DisconnectReason)_reason) << ")";
	if (m_socket.is_open())
	{
		if (m_disconnect == chrono::steady_clock::time_point::max())
		{
			RLPStream s;
			prep(s);
			s.appendList(2) << DisconnectPacket << _reason;
			sealAndSend(s);
			m_disconnect = chrono::steady_clock::now();
		}
		else
			dropped();
	}
}

void Session::start()
{
	RLPStream s;
	prep(s);
	s.appendList(6) << HelloPacket
					<< m_server->protocolVersion()
					<< m_server->m_clientVersion
					<< m_server->caps()
					<< m_server->m_public.port()
					<< m_server->m_id;
	sealAndSend(s);
	ping();
	getPeers();

	doRead();
}

void Session::doRead()
{
	// ignore packets received while waiting to disconnect
	if (chrono::steady_clock::now() - m_disconnect > chrono::seconds(0))
		return;
	
	auto self(shared_from_this());
	m_socket.async_read_some(boost::asio::buffer(m_data), [this,self](boost::system::error_code ec, std::size_t length)
	{
		// If error is end of file, ignore
		if (ec && ec.category() != boost::asio::error::get_misc_category() && ec.value() != boost::asio::error::eof)
		{
			// got here with length of 1241...
			cwarn << "Error reading: " << ec.message();
			dropped();
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
						cwarn << "INVALID SYNCHRONISATION TOKEN; expected = 22400891; received = " << toHex(bytesConstRef(m_incoming.data(), 4));
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
							cwarn << "INVALID MESSAGE RECEIVED";
							disconnect(BadProtocol);
							return;
						}
						else
						{
							RLP r(data.cropped(8));
							if (!interpret(r))
							{
								// error
								dropped();
								return;
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
				dropped();
			}
			catch (std::exception const& _e)
			{
				clogS(NetWarn) << "ERROR: " << _e.what();
				dropped();
			}
		}
	});
}
