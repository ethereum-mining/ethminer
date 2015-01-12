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

#if defined(clogS)
#undef clogS
#endif
#define clogS(X) dev::LogOutputStream<X, true>(false) << "| " << std::setw(2) << session()->socketId() << "] "

EthereumPeer::EthereumPeer(Session* _s, HostCapabilityFace* _h, unsigned _i):
	Capability(_s, _h, _i),
	m_sub(host()->m_man)
{
	transition(Asking::State);
}

EthereumPeer::~EthereumPeer()
{
	abortSync();
}

void EthereumPeer::abortSync()
{
	if (isSyncing())
		transition(Asking::Nothing, true);
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

void EthereumPeer::transition(Asking _a, bool _force)
{
	clogS(NetMessageSummary) << "Transition!" << ::toString(_a) << "from" << ::toString(m_asking) << ", " << (isSyncing() ? "syncing" : "holding") << (needsSyncing() ? "& needed" : "");

	if (m_asking == Asking::State && _a != Asking::State)
		m_requireTransactions = true;

	RLPStream s;

	if (_a == Asking::State)
	{
		if (m_asking == Asking::Nothing)
		{
			setAsking(Asking::State, false);
			prep(s, StatusPacket, 5)
							<< host()->protocolVersion()
							<< host()->networkId()
							<< host()->m_chain.details().totalDifficulty
							<< host()->m_chain.currentHash()
							<< host()->m_chain.genesisHash();
			sealAndSend(s);
			return;
		}
	}
	else if (_a == Asking::Hashes)
	{
		if (m_asking == Asking::State || m_asking == Asking::Nothing)
		{
			if (isSyncing())
				clogS(NetWarn) << "Bad state: not asking for Hashes, yet syncing!";

			m_syncingLatestHash = m_latestHash;
			m_syncingTotalDifficulty = m_totalDifficulty;
			resetNeedsSyncing();

			setAsking(_a, true);
			prep(s, GetBlockHashesPacket, 2) << m_syncingLatestHash << c_maxHashesAsk;
			m_syncingNeededBlocks = h256s(1, m_syncingLatestHash);
			sealAndSend(s);
			return;
		}
		else if (m_asking == Asking::Hashes)
		{
			if (!isSyncing())
				clogS(NetWarn) << "Bad state: asking for Hashes yet not syncing!";

			setAsking(_a, true);
			prep(s, GetBlockHashesPacket, 2) << m_syncingNeededBlocks.back() << c_maxHashesAsk;
			sealAndSend(s);
			return;
		}
	}
	else if (_a == Asking::Blocks)
	{
		if (m_asking == Asking::Hashes)
		{
			if (!isSyncing())
				clogS(NetWarn) << "Bad state: asking for Hashes yet not syncing!";
			if (shouldGrabBlocks())
			{
				clog(NetNote) << "Difficulty of hashchain HIGHER. Grabbing" << m_syncingNeededBlocks.size() << "blocks [latest now" << m_syncingLatestHash.abridged() << ", was" << host()->m_latestBlockSent.abridged() << "]";

				host()->m_man.resetToChain(m_syncingNeededBlocks);
				host()->m_latestBlockSent = m_syncingLatestHash;
			}
			else
			{
				clog(NetNote) << "Difficulty of hashchain not HIGHER. Ignoring.";
				m_syncingLatestHash = h256();
				setAsking(Asking::Nothing, false);
				return;
			}
		}
		// run through into...
		if (m_asking == Asking::Nothing || m_asking == Asking::Hashes || m_asking == Asking::Blocks)
		{
			// Looks like it's the best yet for total difficulty. Set to download.
			setAsking(Asking::Blocks, isSyncing());		// will kick off other peers to help if available.
			auto blocks = m_sub.nextFetch(c_maxBlocksAsk);
			if (blocks.size())
			{
				prep(s, GetBlocksPacket, blocks.size());
				for (auto const& i: blocks)
					s << i;
				sealAndSend(s);
			}
			else
				transition(Asking::Nothing);
			return;
		}
	}
	else if (_a == Asking::Nothing)
	{
		if (m_asking == Asking::Blocks)
		{
			clogS(NetNote) << "Finishing blocks fetch...";

			// a bit overkill given that the other nodes may yet have the needed blocks, but better to be safe than sorry.
			if (isSyncing())
				host()->noteDoneBlocks(this, _force);

			// NOTE: need to notify of giving up on chain-hashes, too, altering state as necessary.
			m_sub.doneFetch();

			setAsking(Asking::Nothing, false);
		}
		else if (m_asking == Asking::Hashes)
		{
			clogS(NetNote) << "Finishing hashes fetch...";

			setAsking(Asking::Nothing, false);
		}
		else if (m_asking == Asking::State)
		{
			setAsking(Asking::Nothing, false);
			// Just got the state - should check to see if we can be of help downloading the chain if any.
			// Otherwise, should put ourselves up for sync.
			setNeedsSyncing(m_latestHash, m_totalDifficulty);
		}
		// Otherwise it's fine. We don't care if it's Nothing->Nothing.
		return;
	}

	clogS(NetWarn) << "Invalid state transition:" << ::toString(_a) << "from" << ::toString(m_asking) << ", " << (isSyncing() ? "syncing" : "holding") << (needsSyncing() ? "& needed" : "");
}

void EthereumPeer::setAsking(Asking _a, bool _isSyncing)
{
	bool changedAsking = (m_asking != _a);
	m_asking = _a;

	if (_isSyncing != (host()->m_syncer == this) || (_isSyncing && changedAsking))
		host()->changeSyncer(_isSyncing ? this : nullptr);

	if (!_isSyncing)
	{
		m_syncingLatestHash = h256();
		m_syncingTotalDifficulty = 0;
		m_syncingNeededBlocks.clear();
	}

	session()->addNote("ask", _a == Asking::Nothing ? "nothing" : _a == Asking::State ? "state" : _a == Asking::Hashes ? "hashes" : _a == Asking::Blocks ? "blocks" : "?");
	session()->addNote("sync", string(isSyncing() ? "ongoing" : "holding") + (needsSyncing() ? " & needed" : ""));
}

void EthereumPeer::setNeedsSyncing(h256 _latestHash, u256 _td)
{
	m_latestHash = _latestHash;
	m_totalDifficulty = _td;

	if (m_latestHash)
		host()->noteNeedsSyncing(this);

	session()->addNote("sync", string(isSyncing() ? "ongoing" : "holding") + (needsSyncing() ? " & needed" : ""));
}

bool EthereumPeer::isSyncing() const
{
	return host()->m_syncer == this;
}

bool EthereumPeer::shouldGrabBlocks() const
{
	auto td = m_syncingTotalDifficulty;
	auto lh = m_syncingLatestHash;
	auto ctd = host()->m_chain.details().totalDifficulty;

	if (m_syncingNeededBlocks.empty())
		return false;

	clog(NetNote) << "Should grab blocks? " << td << "vs" << ctd << ";" << m_syncingNeededBlocks.size() << " blocks, ends" << m_syncingNeededBlocks.back().abridged();

	if (td < ctd || (td == ctd && host()->m_chain.currentHash() == lh))
		return false;

	return true;
}

void EthereumPeer::attemptSync()
{
	if (m_asking != Asking::Nothing)
	{
		clogS(NetAllDetail) << "Can't synced with this peer - outstanding asks.";
		return;
	}

	// if already done this, then ignore.
	if (!needsSyncing())
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
		resetNeedsSyncing();
		transition(Asking::Nothing);
	}
	else
	{
		clogS(NetAllDetail) << "Yes. Their chain is better.";
		transition(Asking::Hashes);
	}
}

bool EthereumPeer::interpret(unsigned _id, RLP const& _r)
{
	try
	{
	switch (_id)
	{
	case StatusPacket:
	{
		m_protocolVersion = _r[1].toInt<unsigned>();
		m_networkId = _r[2].toInt<u256>();

		// a bit dirty as we're misusing these to communicate the values to transition, but harmless.
		m_totalDifficulty = _r[3].toInt<u256>();
		m_latestHash = _r[4].toHash<h256>();
		auto genesisHash = _r[5].toHash<h256>();

		clogS(NetMessageSummary) << "Status:" << m_protocolVersion << "/" << m_networkId << "/" << genesisHash.abridged() << ", TD:" << m_totalDifficulty << "=" << m_latestHash.abridged();

		if (genesisHash != host()->m_chain.genesisHash())
			disable("Invalid genesis hash");
		else if (m_protocolVersion != host()->protocolVersion())
			disable("Invalid protocol version.");
		else if (m_networkId != host()->networkId())
			disable("Invalid network identifier.");
		else if (session()->info().clientVersion.find("/v0.7.0/") != string::npos)
			disable("Blacklisted client version.");
		else if (host()->isBanned(session()->id()))
			disable("Peer banned for previous bad behaviour.");
		else
			transition(Asking::Nothing);
		break;
	}
	case GetTransactionsPacket: break;	// DEPRECATED.
	case TransactionsPacket:
	{
		clogS(NetMessageSummary) << "Transactions (" << dec << (_r.itemCount() - 1) << "entries)";
		addRating(_r.itemCount() - 1);
		Guard l(x_knownTransactions);
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto h = sha3(_r[i].data());
			m_knownTransactions.insert(h);
			if (!host()->m_tq.import(_r[i].data()))
				// if we already had the transaction, then don't bother sending it on.
				host()->m_transactionsSent.insert(h);
		}
		break;
	}
	case GetBlockHashesPacket:
	{
		h256 later = _r[1].toHash<h256>();
		unsigned limit = _r[2].toInt<unsigned>();
		clogS(NetMessageSummary) << "GetBlockHashes (" << limit << "entries," << later.abridged() << ")";

		unsigned c = min<unsigned>(host()->m_chain.number(later), limit);

		RLPStream s;
		prep(s, BlockHashesPacket, c);
		h256 p = host()->m_chain.details(later).parent;
		for (unsigned i = 0; i < c && p; ++i, p = host()->m_chain.details(p).parent)
			s << p;
		sealAndSend(s);
		break;
	}
	case BlockHashesPacket:
	{
		clogS(NetMessageSummary) << "BlockHashes (" << dec << (_r.itemCount() - 1) << "entries)" << (_r.itemCount() - 1 ? "" : ": NoMoreHashes");

		if (m_asking != Asking::Hashes)
		{
			cwarn << "Peer giving us hashes when we didn't ask for them.";
			break;
		}
		if (_r.itemCount() == 1)
		{
			transition(Asking::Blocks);
			return true;
		}
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto h = _r[i].toHash<h256>();
			if (host()->m_chain.isKnown(h))
			{
				transition(Asking::Blocks);
				return true;
			}
			else
				m_syncingNeededBlocks.push_back(h);
		}
		// run through - ask for more.
		transition(Asking::Hashes);
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
		prep(s, BlocksPacket, n).appendRaw(rlp, n);
		sealAndSend(s);
		break;
	}
	case BlocksPacket:
	{
		clogS(NetMessageSummary) << "Blocks (" << dec << (_r.itemCount() - 1) << "entries)" << (_r.itemCount() - 1 ? "" : ": NoMoreBlocks");

		if (m_asking != Asking::Blocks)
			clogS(NetWarn) << "Unexpected Blocks received!";

		if (_r.itemCount() == 1)
		{
			// Got to this peer's latest block - just give up.
			transition(Asking::Nothing);
			break;
		}

		unsigned success = 0;
		unsigned future = 0;
		unsigned unknown = 0;
		unsigned got = 0;
		unsigned repeated = 0;

		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto h = BlockInfo::headerHash(_r[i].data());
			if (m_sub.noteBlock(h))
			{
				addRating(10);
				switch (host()->m_bq.import(_r[i].data(), host()->m_chain))
				{
				case ImportResult::Success:
					success++;
					break;

				case ImportResult::Malformed:
					disable("Malformed block received.");
					return true;

				case ImportResult::FutureTime:
					future++;
					break;

				case ImportResult::AlreadyInChain:
				case ImportResult::AlreadyKnown:
					got++;
					break;

				case ImportResult::UnknownParent:
					unknown++;
					break;
				}
			}
			else
			{
				addRating(0);	// -1?
				repeated++;
			}
		}

		clogS(NetMessageSummary) << dec << success << "imported OK," << unknown << "with unknown parents," << future << "with future timestamps," << got << " already known," << repeated << " repeats received.";

		if (m_asking == Asking::Blocks)
			transition(Asking::Blocks);
		break;
	}
	case NewBlockPacket:
	{
		auto h = BlockInfo::headerHash(_r[1].data());
		clogS(NetMessageSummary) << "NewBlock: " << h.abridged();

		if (_r.itemCount() != 3)
			disable("NewBlock without 2 data fields.");
		else
		{
			switch (host()->m_bq.import(_r[1].data(), host()->m_chain))
			{
			case ImportResult::Success:
				addRating(100);
				break;
			case ImportResult::FutureTime:
				//TODO: Rating dependent on how far in future it is.
				break;

			case ImportResult::Malformed:
				disable("Malformed block received.");
				break;

			case ImportResult::AlreadyInChain:
			case ImportResult::AlreadyKnown:
				break;

			case ImportResult::UnknownParent:
				clogS(NetMessageSummary) << "Received block with no known parent. Resyncing...";
				setNeedsSyncing(h, _r[2].toInt<u256>());
				break;
			}
			Guard l(x_knownBlocks);
			m_knownBlocks.insert(h);
		}
		break;
	}
	default:
		return false;
	}
	}
	catch (std::exception const& _e)
	{
		clogS(NetWarn) << "Peer causing an exception:" << _e.what() << _r;
	}

	return true;
}
