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
#include <libethential/Common.h>
#include <libethcore/Exceptions.h>
#include <libp2p/Session.h>
#include "BlockChain.h"
#include "EthereumHost.h"
using namespace std;
using namespace eth;
using namespace p2p;

#define clogS(X) eth::LogOutputStream<X, true>(false) << "| " << std::setw(2) << session()->socketId() << "] "

EthereumPeer::EthereumPeer(Session* _s, HostCapabilityFace* _h):
	Capability(_s, _h)
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
					<< host()->m_chain->details().totalDifficulty
					<< host()->m_chain->currentHash()
					<< host()->m_chain->genesisHash();
	sealAndSend(s);
}

void EthereumPeer::startInitialSync()
{
	// Grab trsansactions off them.
	{
		RLPStream s;
		prep(s).appendList(1);
		s << GetTransactionsPacket;
		sealAndSend(s);
	}

	h256 c = host()->m_chain->currentHash();
	uint n = host()->m_chain->number();
	u256 td = max(host()->m_chain->details().totalDifficulty, host()->m_totalDifficultyOfNeeded);

	clogS(NetAllDetail) << "Initial sync. Latest:" << c.abridged() << ", number:" << n << ", TD: max(" << host()->m_chain->details().totalDifficulty << "," << host()->m_totalDifficultyOfNeeded << ") versus " << m_totalDifficulty;
	if (td > m_totalDifficulty)
		return;	// All good - we have the better chain.

	// Our chain isn't better - grab theirs.
	{
		RLPStream s;
		prep(s).appendList(3);
		s << GetBlockHashesPacket << m_latestHash << c_maxHashesAsk;
		m_neededBlocks = h256s(1, m_latestHash);
		sealAndSend(s);
	}
}

inline string toString(h256s const& _bs)
{
	ostringstream out;
	out << "[ ";
	for (auto i: _bs)
		out << i.abridged() << ", ";
	out << "]";
	return out.str();
}

void EthereumPeer::giveUpOnFetch()
{
	clogS(NetNote) << "GIVE UP FETCH; can't get" << toString(m_askedBlocks);
	if (m_askedBlocks.size())
	{
		Guard l (host()->x_blocksNeeded);
		host()->m_blocksNeeded.reserve(host()->m_blocksNeeded.size() + m_askedBlocks.size());
		for (auto i: m_askedBlocks)
		{
			m_failedBlocks.insert(i);
			host()->m_blocksOnWay.erase(i);
			host()->m_blocksNeeded.push_back(i);
		}
		m_askedBlocks.clear();
	}
}

bool EthereumPeer::interpret(RLP const& _r)
{
	switch (_r[0].toInt<unsigned>())
	{
	case StatusPacket:
	{
		m_protocolVersion = _r[1].toInt<uint>();
		m_networkId = _r[2].toInt<u256>();
		m_totalDifficulty = _r[3].toInt<u256>();
		m_latestHash = _r[4].toHash<h256>();
		auto genesisHash = _r[5].toHash<h256>();

		clogS(NetMessageSummary) << "Status:" << m_protocolVersion << "/" << m_networkId << "/" << genesisHash.abridged() << ", TD:" << m_totalDifficulty << "=" << m_latestHash.abridged();

		if (genesisHash != host()->m_chain->genesisHash())
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
		clogS(NetMessageSummary) << "Transactions (" << dec << (_r.itemCount() - 1) << "entries)";
		addRating(_r.itemCount() - 1);
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			host()->m_incomingTransactions.push_back(_r[i].data().toBytes());
			m_knownTransactions.insert(sha3(_r[i].data()));
		}
		break;
	case GetBlockHashesPacket:
	{
		h256 later = _r[1].toHash<h256>();
		unsigned limit = _r[2].toInt<unsigned>();
		clogS(NetMessageSummary) << "GetBlockHashes (" << limit << "entries," << later.abridged() << ")";

		unsigned c = min<unsigned>(host()->m_chain->number(later), limit);

		RLPStream s;
		prep(s).appendList(1 + c).append(BlockHashesPacket);
		h256 p = host()->m_chain->details(later).parent;
		for (unsigned i = 0; i < c; ++i, p = host()->m_chain->details(p).parent)
			s << p;
		sealAndSend(s);
		break;
	}
	case BlockHashesPacket:
	{
		clogS(NetMessageSummary) << "BlockHashes (" << dec << (_r.itemCount() - 1) << "entries)";
		if (_r.itemCount() == 1)
		{
			host()->noteHaveChain(this);
			return true;
		}
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto h = _r[i].toHash<h256>();
			if (host()->m_chain->details(h))
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
		// TODO: return the requested blocks.
		bytes rlp;
		unsigned n = 0;
		for (unsigned i = 1; i < _r.itemCount() && i <= c_maxBlocks; ++i)
		{
			auto b = host()->m_chain->block(_r[i].toHash<h256>());
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
		clogS(NetMessageSummary) << "Blocks (" << dec << (_r.itemCount() - 1) << "entries)";

		if (_r.itemCount() == 1 && !m_askedBlocksChanged)
		{
			// Couldn't get any from last batch - probably got to this peer's latest block - just give up.
			giveUpOnFetch();
		}
		m_askedBlocksChanged = false;

		unsigned used = 0;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto h = BlockInfo::headerHash(_r[i].data());
			if (host()->noteBlock(h, _r[i].data()))
				used++;
			m_askedBlocks.erase(h);
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
				if (!host()->m_chain->details(bi.parentHash) && !m_knownBlocks.count(bi.parentHash))
				{
					unknownParents++;
					clogS(NetAllDetail) << "Unknown parent" << bi.parentHash << "of block" << h;
				}
				else
				{
					knownParents++;
					clogS(NetAllDetail) << "Known parent" << bi.parentHash << "of block" << h;
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

void EthereumPeer::restartGettingChain()
{
	if (m_askedBlocks.size())
	{
		m_askedBlocksChanged = true;	// So that we continue even if the Ask's reply is empty.
		m_askedBlocks.clear();			// So that we restart once we get the Ask's reply.
		m_failedBlocks.clear();
	}
	else
		ensureGettingChain();
}

void EthereumPeer::ensureGettingChain()
{
	if (m_askedBlocks.size())
		return;	// Already asked & waiting for some.

	continueGettingChain();
}

void EthereumPeer::continueGettingChain()
{
	if (!m_askedBlocks.size())
		m_askedBlocks = host()->neededBlocks(m_failedBlocks);

	if (m_askedBlocks.size())
	{
		RLPStream s;
		prep(s);
		s.appendList(m_askedBlocks.size() + 1) << GetBlocksPacket;
		for (auto i: m_askedBlocks)
			s << i;
		sealAndSend(s);
	}
	else
	{
		clogS(NetMessageSummary) << "No blocks left to get. Peer doesn't seem to have" << m_failedBlocks.size() << "of our needed blocks.";
		host()->noteDoneBlocks();
	}
}
