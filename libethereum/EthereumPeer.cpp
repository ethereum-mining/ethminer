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
/** @file EthereumPeer.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "EthereumPeer.h"

#include <chrono>
#include <libdevcore/Common.h>
#include <libethcore/Exceptions.h>
#include <libp2p/Session.h>
#include "BlockChain.h"
#include "EthereumHost.h"
#include "TransactionQueue.h"
#include "BlockQueue.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

EthereumPeer::EthereumPeer(Session* _s, HostCapabilityFace* _h, unsigned _i, CapDesc const& _cap):
	Capability(_s, _h, _i),
	m_sub(host()->downloadMan()),
	m_hashSub(host()->hashDownloadMan()),
	m_peerCapabilityVersion(_cap.second)
{
	m_syncHashNumber = host()->chain().number() + 1;
	requestStatus();
}

EthereumPeer::~EthereumPeer()
{
	clog(NetMessageSummary) << "Aborting Sync :-(";
	abortSync();
}

void EthereumPeer::abortSync()
{
	if (isSyncing())
		setIdle();
}

EthereumHost* EthereumPeer::host() const
{
	return static_cast<EthereumHost*>(Capability::hostCapability());
}

/*
 * Possible asking/syncing states for two peers:
 */

string toString(Asking _a)
{
	switch (_a)
	{
	case Asking::Blocks: return "Blocks";
	case Asking::Hashes: return "Hashes";
	case Asking::Nothing: return "Nothing";
	case Asking::State: return "State";
	}
	return "?";
}

void EthereumPeer::setIdle()
{
	m_sub.doneFetch();
	m_hashSub.doneFetch();
	setAsking(Asking::Nothing);
}

void EthereumPeer::requestStatus()
{
	assert(m_asking == Asking::Nothing);
	setAsking(Asking::State);
	RLPStream s;
	bool latest = m_peerCapabilityVersion == host()->protocolVersion();
	prep(s, StatusPacket, latest ? 6 : 5)
					<< (latest ? host()->protocolVersion() : EthereumHost::c_oldProtocolVersion)
					<< host()->networkId()
					<< host()->chain().details().totalDifficulty
					<< host()->chain().currentHash()
					<< host()->chain().genesisHash();
	if (latest)
		s << u256(host()->chain().number());
	sealAndSend(s);
}

void EthereumPeer::requestHashes()
{
	assert(m_asking == Asking::Nothing);
	m_syncHashNumber = m_hashSub.nextFetch(c_maxHashesAsk);
	setAsking(Asking::Hashes);
	RLPStream s;
	prep(s, GetBlockHashesByNumberPacket, 2) << m_syncHashNumber << c_maxHashesAsk;
	clog(NetMessageDetail) << "Requesting block hashes for numbers " << m_syncHashNumber << "-" << m_syncHashNumber + c_maxHashesAsk - 1;
	sealAndSend(s);
}

void EthereumPeer::requestHashes(h256 const& _lastHash)
{
	assert(m_asking == Asking::Nothing);
	setAsking(Asking::Hashes);
	RLPStream s;
	prep(s, GetBlockHashesPacket, 2) << _lastHash << c_maxHashesAsk;
	clog(NetMessageDetail) << "Requesting block hashes staring from " << _lastHash;
	sealAndSend(s);
}

void EthereumPeer::requestBlocks()
{
	setAsking(Asking::Blocks);
	auto blocks = m_sub.nextFetch(c_maxBlocksAsk);
	if (blocks.size())
	{
		RLPStream s;
		prep(s, GetBlocksPacket, blocks.size());
		for (auto const& i: blocks)
			s << i;
		sealAndSend(s);
	}
	else
		setIdle();
	return;
}

void EthereumPeer::setAsking(Asking _a)
{
	m_asking = _a;
	m_lastAsk = chrono::system_clock::now();

	session()->addNote("ask", _a == Asking::Nothing ? "nothing" : _a == Asking::State ? "state" : _a == Asking::Hashes ? "hashes" : _a == Asking::Blocks ? "blocks" : "?");
	session()->addNote("sync", string(isSyncing() ? "ongoing" : "holding") + (needsSyncing() ? " & needed" : ""));
}

void EthereumPeer::tick()
{
	if (chrono::system_clock::now() - m_lastAsk > chrono::seconds(10) && m_asking != Asking::Nothing)
		// timeout
		session()->disconnect(PingTimeout);
}

bool EthereumPeer::isSyncing() const
{
	return m_asking != Asking::Nothing;
}

bool EthereumPeer::interpret(unsigned _id, RLP const& _r)
{
	try
	{
	switch (_id)
	{
	case StatusPacket:
	{
		m_protocolVersion = _r[0].toInt<unsigned>();
		m_networkId = _r[1].toInt<u256>();
		m_totalDifficulty = _r[2].toInt<u256>();
		m_latestHash = _r[3].toHash<h256>();
		m_genesisHash = _r[4].toHash<h256>();
		if (m_peerCapabilityVersion == host()->protocolVersion())
		{
			m_protocolVersion = host()->protocolVersion();
			m_latestBlockNumber = _r[5].toInt<u256>();
		}

		clog(NetMessageSummary) << "Status:" << m_protocolVersion << "/" << m_networkId << "/" << m_genesisHash << "/" << m_latestBlockNumber << ", TD:" << m_totalDifficulty << "=" << m_latestHash;
		setAsking(Asking::Nothing);
		host()->onPeerStatus(this);
		break;
	}
	case TransactionsPacket:
	{
		host()->onPeerTransactions(this, _r);
		break;
	}
	case GetBlockHashesPacket:
	{
		h256 later = _r[0].toHash<h256>();
		unsigned limit = _r[1].toInt<unsigned>();
		clog(NetMessageSummary) << "GetBlockHashes (" << limit << "entries," << later << ")";
		unsigned c = min<unsigned>(host()->chain().number(later), limit);
		RLPStream s;
		prep(s, BlockHashesPacket, c);
		h256 p = host()->chain().details(later).parent;
		for (unsigned i = 0; i < c && p; ++i, p = host()->chain().details(p).parent)
			s << p;
		sealAndSend(s);
		addRating(0);
		break;
	}
	case GetBlockHashesByNumberPacket:
	{
		u256 number256 = _r[0].toInt<u256>();
		unsigned number = (unsigned) number256;
		unsigned limit = _r[1].toInt<unsigned>();
		clog(NetMessageSummary) << "GetBlockHashesByNumber (" << number << "-" << number + limit - 1 << ")";
		RLPStream s;
		if (number <= host()->chain().number())
		{
			unsigned c = min<unsigned>(host()->chain().number() - number + 1, limit);
			prep(s, BlockHashesPacket, c);
			for (unsigned n = number; n < number + c; n++)
			{
				h256 p = host()->chain().numberHash(n);
				s << p;
			}
		}
		else
			prep(s, BlockHashesPacket, 0);
		sealAndSend(s);
		addRating(0);
		break;
	}
	case BlockHashesPacket:
	{
		unsigned itemCount = _r.itemCount();
		clog(NetMessageSummary) << "BlockHashes (" << dec << itemCount << "entries)" << (itemCount ? "" : ": NoMoreHashes");

		if (m_asking != Asking::Hashes)
		{
			clog(NetWarn) << "Peer giving us hashes when we didn't ask for them.";
			break;
		}
		setAsking(Asking::Nothing);
		h256s hashes(itemCount);
		for (unsigned i = 0; i < itemCount; ++i)
		{
			hashes[i] = _r[i].toHash<h256>();
			m_hashSub.noteHash(m_syncHashNumber + i, 1);
		}

		m_syncHashNumber += itemCount;
		host()->onPeerHashes(this, hashes);
		break;
	}
	case GetBlocksPacket:
	{
		unsigned count = _r.itemCount();
		clog(NetMessageSummary) << "GetBlocks (" << dec << count << "entries)";

		if (!count)
		{
			clog(NetImpolite) << "Zero-entry GetBlocks: Not replying.";
			addRating(-10);
			break;
		}
		// return the requested blocks.
		bytes rlp;
		unsigned n = 0;
		for (unsigned i = 0; i < min(count, c_maxBlocks); ++i)
		{
			auto h = _r[i].toHash<h256>();
			if (host()->chain().isKnown(h))
			{
				rlp += host()->chain().block(_r[i].toHash<h256>());
				++n;
			}
		}
		if (count > 20 && n == 0)
			clog(NetWarn) << "all" << count << "unknown blocks requested; peer on different chain?";
		else
			clog(NetMessageSummary) << n << "blocks known and returned;" << (min(count, c_maxBlocks) - n) << "blocks unknown;" << (count > c_maxBlocks ? count - c_maxBlocks : 0) << "blocks ignored";

		addRating(0);
		RLPStream s;
		prep(s, BlocksPacket, n).appendRaw(rlp, n);
		sealAndSend(s);
		break;
	}
	case BlocksPacket:
	{
		if (m_asking != Asking::Blocks)
			clog(NetWarn) << "Peer giving us blocks when we didn't ask for them.";
		else
		{
			setAsking(Asking::Nothing);
			host()->onPeerBlocks(this, _r);
		}
		break;
	}
	case NewBlockPacket:
	{
		host()->onPeerNewBlock(this, _r);
		break;
	}
	case NewBlockHashesPacket:
	{
		unsigned itemCount = _r.itemCount();
		clog(NetMessageSummary) << "BlockHashes (" << dec << itemCount << "entries)" << (itemCount ? "" : ": NoMoreHashes");

		h256s hashes(itemCount);
		for (unsigned i = 0; i < itemCount; ++i)
			hashes[i] = _r[i].toHash<h256>();

		host()->onPeerNewHashes(this, hashes);
		break;
	}
	default:
		return false;
	}
	}
	catch (Exception const& _e)
	{
		clog(NetWarn) << "Peer causing an Exception:" << _e.what() << _r;
	}
	catch (std::exception const& _e)
	{
		clog(NetWarn) << "Peer causing an exception:" << _e.what() << _r;
	}

	return true;
}
