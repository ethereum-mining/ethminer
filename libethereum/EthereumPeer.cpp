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
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

#define clogS(X) dev::LogOutputStream<X, true>(false) << "| " << std::setw(2) << session()->socketId() << "] "

EthereumPeer::EthereumPeer(Session* _s, HostCapabilityFace* _h):
	Capability(_s, _h),
	m_sub(host()->m_man)
{
	sendStatus();
}

EthereumPeer::~EthereumPeer()
{
	giveUpOnFetch();
}

EthereumHost* EthereumPeer::host() const
{
	return static_cast<EthereumHost*>(Capability::hostCapability());
}

void EthereumPeer::sendStatus()
{
	RLPStream s;
	prep(s);
	s.appendList(6) << StatusPacket
					<< host()->protocolVersion()
					<< host()->networkId()
					<< host()->m_chain.details().totalDifficulty
					<< host()->m_chain.currentHash()
					<< host()->m_chain.genesisHash();
	sealAndSend(s);
}

void EthereumPeer::startInitialSync()
{
	// Grab transactions off them.
	{
		RLPStream s;
		prep(s).appendList(1);
		s << GetTransactionsPacket;
		sealAndSend(s);
	}

	host()->noteHavePeerState(this);
}

void EthereumPeer::tryGrabbingHashChain()
{
	// if already done this, then ignore.
	if (m_grabbing != Grabbing::State)
	{
		clogS(NetAllDetail) << "Already synced with this peer.";
		return;
	}

	h256 c = host()->m_chain.currentHash();
	unsigned n = host()->m_chain.number();
	u256 td = host()->m_chain.details().totalDifficulty;

	clogS(NetAllDetail) << "Attempt chain-grab? Latest:" << c.abridged() << ", number:" << n << ", TD:" << td << " versus " << m_totalDifficulty;
	if (td >= m_totalDifficulty)
	{
		clogS(NetAllDetail) << "No. Our chain is better.";
		m_grabbing = Grabbing::Nothing;
		return;	// All good - we have the better chain.
	}

	// Our chain isn't better - grab theirs.
	{
		clogS(NetAllDetail) << "Yes. Their chain is better.";

		host()->updateGrabbing(Grabbing::Hashes);
		m_grabbing = Grabbing::Hashes;
		RLPStream s;
		prep(s).appendList(3);
		s << GetBlockHashesPacket << m_latestHash << c_maxHashesAsk;
		m_neededBlocks = h256s(1, m_latestHash);
		sealAndSend(s);
	}
}

void EthereumPeer::giveUpOnFetch()
{
	clogS(NetNote) << "GIVE UP FETCH";

	// a bit overkill given that the other nodes may yet have the needed blocks, but better to be safe than sorry.
	if (m_grabbing == Grabbing::Chain || m_grabbing == Grabbing::ChainHelper)
	{
		host()->noteDoneBlocks(this);
		m_grabbing = Grabbing::Nothing;
	}

	m_sub.doneFetch();
	// NOTE: need to notify of giving up on chain-hashes, too, altering state as necessary.
}

bool EthereumPeer::interpret(RLP const& _r)
{
	switch (_r[0].toInt<unsigned>())
	{
	case StatusPacket:
	{
		m_protocolVersion = _r[1].toInt<unsigned>();
		m_networkId = _r[2].toInt<u256>();
		m_totalDifficulty = _r[3].toInt<u256>();
		m_latestHash = _r[4].toHash<h256>();
		auto genesisHash = _r[5].toHash<h256>();

		clogS(NetMessageSummary) << "Status:" << m_protocolVersion << "/" << m_networkId << "/" << genesisHash.abridged() << ", TD:" << m_totalDifficulty << "=" << m_latestHash.abridged();

		if (genesisHash != host()->m_chain.genesisHash())
			disable("Invalid genesis hash");
		if (m_protocolVersion != host()->protocolVersion())
			disable("Invalid protocol version.");
		if (m_networkId != host()->networkId())
			disable("Invalid network identifier.");

		startInitialSync();
		break;
	}
	case GetTransactionsPacket:
	{
		m_requireTransactions = true;
		break;
	}
	case TransactionsPacket:
	{
		clogS(NetMessageSummary) << "Transactions (" << dec << (_r.itemCount() - 1) << "entries)";
		addRating(_r.itemCount() - 1);
		lock_guard<recursive_mutex> l(host()->m_incomingLock);
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			host()->addIncomingTransaction(_r[i].data().toBytes());
			
			lock_guard<mutex> l(x_knownTransactions);
			m_knownTransactions.insert(sha3(_r[i].data()));
		}
		break;
	}
	case GetBlockHashesPacket:
	{
		h256 later = _r[1].toHash<h256>();
		unsigned limit = _r[2].toInt<unsigned>();
		clogS(NetMessageSummary) << "GetBlockHashes (" << limit << "entries," << later.abridged() << ")";

		unsigned c = min<unsigned>(max<unsigned>(1, host()->m_chain.number(later)) - 1, limit);

		RLPStream s;
		prep(s).appendList(1 + c).append(BlockHashesPacket);
		h256 p = host()->m_chain.details(later).parent;
		for (unsigned i = 0; i < c; ++i, p = host()->m_chain.details(p).parent)
			s << p;
		sealAndSend(s);
		break;
	}
	case BlockHashesPacket:
	{
		clogS(NetMessageSummary) << "BlockHashes (" << dec << (_r.itemCount() - 1) << "entries)" << (_r.itemCount() - 1 ? "" : ": NoMoreHashes");

		if (m_grabbing != Grabbing::Hashes)
		{
			cwarn << "Peer giving us hashes when we didn't ask for them.";
			break;
		}
		if (_r.itemCount() == 1)
		{
			host()->noteHaveChain(this);
			return true;
		}
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto h = _r[i].toHash<h256>();
			if (host()->m_chain.details(h))
			{
				host()->noteHaveChain(this);
				return true;
			}
			else
				m_neededBlocks.push_back(h);
		}
		// run through - ask for more.
		RLPStream s;
		prep(s).appendList(3);
		s << GetBlockHashesPacket << m_neededBlocks.back() << c_maxHashesAsk;
		sealAndSend(s);
		break;
	}
	case GetBlocksPacket:
	{
		clogS(NetMessageSummary) << "GetBlocks (" << dec << (_r.itemCount() - 1) << "entries)";
		// return the requested blocks.
		bytes rlp;
		unsigned n = 0;
		for (unsigned i = 1; i < _r.itemCount() && i <= c_maxBlocks; ++i)
		{
			auto b = host()->m_chain.block(_r[i].toHash<h256>());
			if (b.size())
			{
				rlp += b;
				++n;
			}
		}
		RLPStream s;
		sealAndSend(prep(s).appendList(n + 1).append(BlocksPacket).appendRaw(rlp, n));
		break;
	}
	case BlocksPacket:
	{
		clogS(NetMessageSummary) << "Blocks (" << dec << (_r.itemCount() - 1) << "entries)" << (_r.itemCount() - 1 ? "" : ": NoMoreBlocks");

		if (_r.itemCount() == 1)
		{
			// Couldn't get any from last batch - probably got to this peer's latest block - just give up.
			m_sub.doneFetch();
			giveUpOnFetch();
			break;
		}

		unsigned used = 0;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto h = BlockInfo::headerHash(_r[i].data());
			m_sub.noteBlock(h);
			if (host()->noteBlock(h, _r[i].data()))
				used++;
			Guard l(x_knownBlocks);
			m_knownBlocks.insert(h);
		}
		addRating(used);
		unsigned knownParents = 0;
		unsigned unknownParents = 0;
		if (g_logVerbosity >= NetMessageSummary::verbosity)
		{
			for (unsigned i = 1; i < _r.itemCount(); ++i)
			{
				auto h = BlockInfo::headerHash(_r[i].data());
				BlockInfo bi(_r[i].data());
				Guard l(x_knownBlocks);
				if (!host()->m_chain.details(bi.parentHash) && !m_knownBlocks.count(bi.parentHash))
				{
					unknownParents++;
					clogS(NetAllDetail) << "Unknown parent" << bi.parentHash.abridged() << "of block" << h.abridged();
				}
				else
				{
					knownParents++;
					clogS(NetAllDetail) << "Known parent" << bi.parentHash.abridged() << "of block" << h.abridged();
				}
			}
		}
		clogS(NetMessageSummary) << dec << knownParents << "known parents," << unknownParents << "unknown," << used << "used.";
		continueGettingChain();
		break;
	}
	default:
		return false;
	}
	return true;
}

void EthereumPeer::ensureGettingChain()
{
	if (m_grabbing == Grabbing::ChainHelper)
		return;	// Already asked & waiting for some.

	continueGettingChain();
}

void EthereumPeer::continueGettingChain()
{
	if (m_grabbing != Grabbing::Chain)
		m_grabbing = Grabbing::ChainHelper;

	auto blocks = m_sub.nextFetch(c_maxBlocksAsk);

	if (blocks.size())
	{
		RLPStream s;
		prep(s);
		s.appendList(blocks.size() + 1) << GetBlocksPacket;
		for (auto const& i: blocks)
			s << i;
		sealAndSend(s);
	}
	else
		giveUpOnFetch();
}
