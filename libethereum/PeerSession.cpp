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
/** @file PeerSession.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "PeerSession.h"

#include <chrono>
#include <libethential/Common.h>
#include <libethcore/Exceptions.h>
#include "BlockChain.h"
#include "PeerServer.h"
using namespace std;
using namespace eth;

#define clogS(X) eth::LogOutputStream<X, true>(false) << "| " << std::setw(2) << m_socket.native_handle() << "] "

static const eth::uint c_maxHashes = 4096;		///< Maximum number of hashes GetChain will ever send.
static const eth::uint c_maxBlocks = 2048;		///< Maximum number of blocks Blocks will ever send.
static const eth::uint c_maxBlocksAsk = 512;	///< Maximum number of blocks we ask to receive in Blocks (when using GetChain).

PeerSession::PeerSession(PeerServer* _s, bi::tcp::socket _socket, u256 _rNId, bi::address _peerAddress, unsigned short _peerPort):
	m_server(_s),
	m_socket(std::move(_socket)),
	m_reqNetworkId(_rNId),
	m_listenPort(_peerPort),
	m_rating(0)
{
	m_disconnect = std::chrono::steady_clock::time_point::max();
	m_connect = std::chrono::steady_clock::now();
	m_info = PeerInfo({"?", _peerAddress.to_string(), m_listenPort, std::chrono::steady_clock::duration(0)});
}

PeerSession::~PeerSession()
{
	// Read-chain finished for one reason or another.
	try
	{
		if (m_socket.is_open())
			m_socket.close();
	}
	catch (...){}
}

bi::tcp::endpoint PeerSession::endpoint() const
{
	if (m_socket.is_open())
		try
		{
			return bi::tcp::endpoint(m_socket.remote_endpoint().address(), m_listenPort);
		}
		catch (...){}

	return bi::tcp::endpoint();
}

bool PeerSession::interpret(RLP const& _r)
{
	clogS(NetRight) << _r;
	switch (_r[0].toInt<unsigned>())
	{
	case HelloPacket:
	{
		m_protocolVersion = _r[1].toInt<uint>();
		m_networkId = _r[2].toInt<u256>();
		auto clientVersion = _r[3].toString();
		m_caps = _r[4].toInt<uint>();
		m_listenPort = _r[5].toInt<unsigned short>();
		m_id = _r[6].toHash<h512>();

		clogS(NetMessageSummary) << "Hello: " << clientVersion << "V[" << m_protocolVersion << "/" << m_networkId << "]" << m_id.abridged() << showbase << hex << m_caps << dec << m_listenPort;

		if (m_server->havePeer(m_id))
		{
			// Already connected.
			cwarn << "Already have peer id" << m_id.abridged();// << "at" << l->endpoint() << "rather than" << endpoint();
			disconnect(DuplicatePeer);
			return false;
		}

		if (m_protocolVersion != PeerServer::protocolVersion() || m_networkId != m_server->networkId() || !m_id)
		{
			disconnect(IncompatibleProtocol);
			return false;
		}
		try
			{ m_info = PeerInfo({clientVersion, m_socket.remote_endpoint().address().to_string(), m_listenPort, std::chrono::steady_clock::duration()}); }
		catch (...)
		{
			disconnect(BadProtocol);
			return false;
		}

		m_server->registerPeer(shared_from_this());
		startInitialSync();

		// Grab trsansactions off them.
		{
			RLPStream s;
			prep(s).appendList(1);
			s << GetTransactionsPacket;
			sealAndSend(s);
		}
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
            clogS(NetTriviaDetail) << "Sending peer " << toHex(i.first.ref().cropped(0, 4)) << i.second;
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
			Public id = _r[i][2].toHash<Public>();
			if (isPrivateAddress(peerAddress))
				goto CONTINUE;

			clogS(NetAllDetail) << "Checking: " << ep << "(" << toHex(id.ref().cropped(0, 4)) << ")";

			// check that it's not us or one we already know:
			if (id && (m_server->m_key.pub() == id || m_server->havePeer(id) || m_server->m_incomingPeers.count(id)))
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
            clogS(NetTriviaDetail) << "New peer: " << ep << "(" << id << ")";
			CONTINUE:;
		}
		break;
	case TransactionsPacket:
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		clogS(NetMessageSummary) << "Transactions (" << dec << (_r.itemCount() - 1) << " entries)";
		m_rating += _r.itemCount() - 1;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			m_server->m_incomingTransactions.push_back(_r[i].data().toBytes());
			m_knownTransactions.insert(sha3(_r[i].data()));
		}
		break;
	case BlocksPacket:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		clogS(NetMessageSummary) << "Blocks (" << dec << (_r.itemCount() - 1) << " entries)";
		unsigned used = 0;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto h = sha3(_r[i].data());
			if (m_server->noteBlock(h, _r[i].data()))
			{
				m_knownBlocks.insert(h);
				used++;
			}
		}
		m_rating += used;
		unsigned knownParents = 0;
		unsigned unknownParents = 0;
		if (g_logVerbosity >= 2)
		{
			for (unsigned i = 1; i < _r.itemCount(); ++i)
			{
				auto h = sha3(_r[i].data());
				BlockInfo bi(_r[i].data());
				if (!m_server->m_chain->details(bi.parentHash) && !m_knownBlocks.count(bi.parentHash))
				{
					unknownParents++;
					clogS(NetMessageDetail) << "Unknown parent " << bi.parentHash << " of block " << h;
				}
				else
				{
					knownParents++;
					clogS(NetMessageDetail) << "Known parent " << bi.parentHash << " of block " << h;
				}
			}
		}
		clogS(NetMessageSummary) << dec << knownParents << " known parents, " << unknownParents << "unknown, " << used << "used.";
		if (used)	// we received some - check if there's any more
		{
			RLPStream s;
			prep(s).appendList(3);
			s << GetChainPacket;
			s << sha3(_r[1].data());
			s << c_maxBlocksAsk;
			sealAndSend(s);
		}
		else
			clogS(NetMessageSummary) << "Peer sent all blocks in chain.";
		break;
	}
	case GetChainPacket:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		clogS(NetMessageSummary) << "GetChain (" << (_r.itemCount() - 2) << " hashes, " << (_r[_r.itemCount() - 1].toInt<bigint>()) << ")";
		// ********************************************************************
		// NEEDS FULL REWRITE!
		h256s parents;
		parents.reserve(_r.itemCount() - 2);
		for (unsigned i = 1; i < _r.itemCount() - 1; ++i)
			parents.push_back(_r[i].toHash<h256>());
		if (_r.itemCount() == 2)
			break;
		// return 2048 block max.
		uint baseCount = (uint)min<bigint>(_r[_r.itemCount() - 1].toInt<bigint>(), c_maxBlocks);
		clogS(NetMessageSummary) << "GetChain (" << baseCount << " max, from " << parents.front() << " to " << parents.back() << ")";
		for (auto parent: parents)
		{
			auto h = m_server->m_chain->currentHash();
			h256 latest = m_server->m_chain->currentHash();
			uint latestNumber = 0;
			uint parentNumber = 0;
			RLPStream s;

			// try to find parent in our blockchain
			// todo: add some delta() fn to blockchain
			BlockDetails fParent = m_server->m_chain->details(parent);
			if (fParent)
			{
				latestNumber = m_server->m_chain->number(latest);
				parentNumber = fParent.number;
				uint count = min<uint>(latestNumber - parentNumber, baseCount);
				clogS(NetAllDetail) << "Requires " << dec << (latestNumber - parentNumber) << " blocks from " << latestNumber << " to " << parentNumber;
				clogS(NetAllDetail) << latest << " - " << parent;

				prep(s);
				s.appendList(1 + count) << BlocksPacket;
				uint endNumber = parentNumber;
				uint startNumber = endNumber + count;
				clogS(NetAllDetail) << "Sending " << dec << count << " blocks from " << startNumber << " to " << endNumber;

				// append blocks
				uint n = latestNumber;
				// seek back (occurs when count is limited by baseCount)
				for (; n > startNumber; n--, h = m_server->m_chain->details(h).parent) {}
				for (uint i = 0; i < count; ++i, --n, h = m_server->m_chain->details(h).parent)
				{
					if (h == parent || n == endNumber)
					{
						cwarn << "BUG! Couldn't create the reply for GetChain!";
						return true;
					}
					clogS(NetAllDetail) << "   " << dec << i << " " << h;
					s.appendRaw(m_server->m_chain->block(h));
				}

				if (!count)
					clogS(NetMessageSummary) << "Sent peer all we have.";
				clogS(NetAllDetail) << "Parent: " << h;
			}
			else if (parent != parents.back())
				continue;

			if (h != parent)
			{
				// not in the blockchain;
				if (parent == parents.back())
				{
					// out of parents...
					clogS(NetAllDetail) << "GetChain failed; not in chain";
					// No good - must have been on a different branch.
					s.clear();
					prep(s).appendList(2) << NotInChainPacket << parents.back();
				}
				else
					// still some parents left - try them.
					continue;
			}
			// send the packet (either Blocks or NotInChain) & exit.
			sealAndSend(s);
			break;
			// ********************************************************************
		}
		break;
	}
	case NotInChainPacket:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		h256 noGood = _r[1].toHash<h256>();
		clogS(NetMessageSummary) << "NotInChain (" << noGood << ")";
		if (noGood == m_server->m_chain->genesisHash())
		{
			clogS(NetWarn) << "Discordance over genesis block! Disconnect.";
			disconnect(WrongGenesis);
		}
		else
		{
			uint count = std::min(c_maxHashes, m_server->m_chain->number(noGood));
			RLPStream s;
			prep(s).appendList(2 + count);
			s << GetChainPacket;
			auto h = m_server->m_chain->details(noGood).parent;
			for (uint i = 0; i < count; ++i, h = m_server->m_chain->details(h).parent)
				s << h;
			s << c_maxBlocksAsk;
			sealAndSend(s);
		}
		break;
	}
	case GetTransactionsPacket:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		m_requireTransactions = true;
		break;
	}
	default:
		break;
	}
	return true;
}

void PeerSession::ping()
{
	RLPStream s;
	sealAndSend(prep(s).appendList(1) << PingPacket);
	m_ping = std::chrono::steady_clock::now();
}

RLPStream& PeerSession::prep(RLPStream& _s)
{
	return _s.appendRaw(bytes(8, 0));
}

void PeerSession::sealAndSend(RLPStream& _s)
{
	bytes b;
	_s.swapOut(b);
	m_server->seal(b);
	sendDestroy(b);
}

bool PeerSession::checkPacket(bytesConstRef _msg)
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

void PeerSession::sendDestroy(bytes& _msg)
{
	clogS(NetLeft) << RLP(bytesConstRef(&_msg).cropped(8));

	if (!checkPacket(bytesConstRef(&_msg)))
	{
		cwarn << "INVALID PACKET CONSTRUCTED!";
	}

	bytes buffer = bytes(std::move(_msg));
	writeImpl(buffer);
}

void PeerSession::send(bytesConstRef _msg)
{
	clogS(NetLeft) << RLP(_msg.cropped(8));
	
	if (!checkPacket(_msg))
	{
		cwarn << "INVALID PACKET CONSTRUCTED!";
	}

	bytes buffer = bytes(_msg.toBytes());
	writeImpl(buffer);
}

void PeerSession::writeImpl(bytes& _buffer)
{
//	cerr << (void*)this << " writeImpl" << endl;
	if (!m_socket.is_open())
		return;

	lock_guard<recursive_mutex> l(m_writeLock);
	m_writeQueue.push_back(_buffer);
	if (m_writeQueue.size() == 1)
		write();
}

void PeerSession::write()
{
//	cerr << (void*)this << " write" << endl;
	lock_guard<recursive_mutex> l(m_writeLock);
	if (m_writeQueue.empty())
		return;
	
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
		}
		else
		{
			m_writeQueue.pop_front();
			write();
		}
	});
}

void PeerSession::dropped()
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

void PeerSession::disconnect(int _reason)
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

void PeerSession::start()
{
	RLPStream s;
	prep(s);
	s.appendList(7) << HelloPacket << (uint)PeerServer::protocolVersion() << m_server->networkId() << m_server->m_clientVersion << (m_server->m_mode == NodeMode::Full ? 0x07 : m_server->m_mode == NodeMode::PeerServer ? 0x01 : 0) << m_server->m_public.port() << m_server->m_key.pub();
	sealAndSend(s);

	ping();

	doRead();
}

void PeerSession::startInitialSync()
{
	uint n = m_server->m_chain->number(m_server->m_latestBlockSent);
	clogS(NetAllDetail) << "Want chain. Latest:" << m_server->m_latestBlockSent << ", number:" << n;
	uint count = std::min(c_maxHashes, n + 1);
	RLPStream s;
	prep(s).appendList(2 + count);
	s << GetChainPacket;
	auto h = m_server->m_latestBlockSent;
	for (uint i = 0; i < count; ++i, h = m_server->m_chain->details(h).parent)
	{
		clogS(NetAllDetail) << "   " << i << ":" << h;
		s << h;
	}

	s << c_maxBlocksAsk;
	sealAndSend(s);
}

void PeerSession::doRead()
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
						doRead();
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
				clogS(NetWarn) << "ERROR: " << _e.description();
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
