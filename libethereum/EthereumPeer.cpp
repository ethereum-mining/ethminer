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

EthereumPeer::EthereumPeer(Session* _s, HostCapabilityFace* _h, unsigned _i):
	Capability(_s, _h, _i),
	m_sub(host()->m_man)
{
	transition(Asking::State);
}

EthereumPeer::~EthereumPeer()
{
	clog(NetMessageSummary) << "Aborting Sync :-(";
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

void EthereumPeer::transition(Asking _a, bool _force, bool _needHelp)
{
	clog(NetMessageSummary) << "Transition!" << ::toString(_a) << "from" << ::toString(m_asking) << ", " << (isSyncing() ? "syncing" : "holding") << (needsSyncing() ? "& needed" : "");

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
				clog(NetWarn) << "Bad state: not asking for Hashes, yet syncing!";

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
				clog(NetWarn) << "Bad state: asking for Hashes yet not syncing!";

			setAsking(_a, true);
			prep(s, GetBlockHashesPacket, 2) << m_syncingLastReceivedHash << c_maxHashesAsk;
			sealAndSend(s);
			return;
		}
	}
	else if (_a == Asking::Blocks)
	{
		if (m_asking == Asking::Hashes)
		{
			if (!isSyncing())
				clog(NetWarn) << "Bad state: asking for Hashes yet not syncing!";
			if (shouldGrabBlocks())
			{
				clog(NetNote) << "Difficulty of hashchain HIGHER. Grabbing" << m_syncingNeededBlocks.size() << "blocks [latest now" << m_syncingLatestHash << ", was" << host()->m_latestBlockSent << "]";

				host()->m_man.resetToChain(m_syncingNeededBlocks);
//				host()->m_latestBlockSent = m_syncingLatestHash;
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
			setAsking(Asking::Blocks, isSyncing(), _needHelp);		// will kick off other peers to help if available.
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
			clog(NetNote) << "Finishing blocks fetch...";

			// a bit overkill given that the other nodes may yet have the needed blocks, but better to be safe than sorry.
			if (isSyncing())
				host()->noteDoneBlocks(this, _force);

			// NOTE: need to notify of giving up on chain-hashes, too, altering state as necessary.
			m_sub.doneFetch();

			setAsking(Asking::Nothing, false);
		}
		else if (m_asking == Asking::Hashes)
		{
			clog(NetNote) << "Finishing hashes fetch...";

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

	clog(NetWarn) << "Invalid state transition:" << ::toString(_a) << "from" << ::toString(m_asking) << ", " << (isSyncing() ? "syncing" : "holding") << (needsSyncing() ? "& needed" : "");
}

void EthereumPeer::setAsking(Asking _a, bool _isSyncing, bool _needHelp)
{
	bool changedAsking = (m_asking != _a);
	m_asking = _a;

	if (_isSyncing != (host()->m_syncer == this) || (_isSyncing && changedAsking))
		host()->changeSyncer(_isSyncing ? this : nullptr, _needHelp);

	if (!_isSyncing)
	{
		m_syncingLatestHash = h256();
		m_syncingTotalDifficulty = 0;
		m_syncingNeededBlocks.clear();
	}

	m_lastAsk = chrono::system_clock::now();

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

void EthereumPeer::tick()
{
	if (chrono::system_clock::now() - m_lastAsk > chrono::seconds(10) && m_asking != Asking::Nothing)
		// timeout
		session()->disconnect(PingTimeout);
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

	clog(NetNote) << "Should grab blocks? " << td << "vs" << ctd << ";" << m_syncingNeededBlocks.size() << " blocks, ends" << m_syncingNeededBlocks.back();

	if (td < ctd || (td == ctd && host()->m_chain.currentHash() == lh))
		return false;

	return true;
}

void EthereumPeer::attemptSync()
{
	if (m_asking != Asking::Nothing)
	{
		clog(NetAllDetail) << "Can't synced with this peer - outstanding asks.";
		return;
	}

	// if already done this, then ignore.
	if (!needsSyncing())
	{
		clog(NetAllDetail) << "Already synced with this peer.";
		return;
	}

	h256 c = host()->m_chain.currentHash();
	unsigned n = host()->m_chain.number();
	u256 td = host()->m_chain.details().totalDifficulty;

	clog(NetAllDetail) << "Attempt chain-grab? Latest:" << c << ", number:" << n << ", TD:" << td << " versus " << m_totalDifficulty;
	if (td >= m_totalDifficulty)
	{
		clog(NetAllDetail) << "No. Our chain is better.";
		resetNeedsSyncing();
		transition(Asking::Nothing);
	}
	else
	{
		clog(NetAllDetail) << "Yes. Their chain is better.";
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
		m_protocolVersion = _r[0].toInt<unsigned>();
		m_networkId = _r[1].toInt<u256>();

		// a bit dirty as we're misusing these to communicate the values to transition, but harmless.
		m_totalDifficulty = _r[2].toInt<u256>();
		m_latestHash = _r[3].toHash<h256>();
		auto genesisHash = _r[4].toHash<h256>();

		clog(NetMessageSummary) << "Status:" << m_protocolVersion << "/" << m_networkId << "/" << genesisHash << ", TD:" << m_totalDifficulty << "=" << m_latestHash;

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
	case TransactionsPacket:
	{
		unsigned itemCount = _r.itemCount();
		clog(NetAllDetail) << "Transactions (" << dec << itemCount << "entries)";
		Guard l(x_knownTransactions);
		for (unsigned i = 0; i < itemCount; ++i)
		{
			auto h = sha3(_r[i].data());
			m_knownTransactions.insert(h);
			ImportResult ir = host()->m_tq.import(_r[i].data());
			switch (ir)
			{
			case ImportResult::Malformed:
				addRating(-100);
				break;
			case ImportResult::AlreadyKnown:
				// if we already had the transaction, then don't bother sending it on.
				host()->m_transactionsSent.insert(h);
				addRating(0);
				break;
			case ImportResult::Success:
				addRating(100);
				break;
			default:;
			}
		}
		break;
	}
	case GetBlockHashesPacket:
	{
		h256 later = _r[0].toHash<h256>();
		unsigned limit = _r[1].toInt<unsigned>();
		clog(NetMessageSummary) << "GetBlockHashes (" << limit << "entries," << later << ")";

		unsigned c = min<unsigned>(host()->m_chain.number(later), limit);

		RLPStream s;
		prep(s, BlockHashesPacket, c);
		h256 p = host()->m_chain.details(later).parent;
		for (unsigned i = 0; i < c && p; ++i, p = host()->m_chain.details(p).parent)
			s << p;
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
			cwarn << "Peer giving us hashes when we didn't ask for them.";
			break;
		}
		if (itemCount == 0)
		{
			transition(Asking::Blocks);
			return true;
		}
		unsigned knowns = 0;
		unsigned unknowns = 0;
		for (unsigned i = 0; i < itemCount; ++i)
		{
			addRating(1);
			auto h = _r[i].toHash<h256>();
			auto status = host()->m_bq.blockStatus(h);
			if (status == QueueStatus::Importing || status == QueueStatus::Ready || host()->m_chain.isKnown(h))
			{
				clog(NetMessageSummary) << "block hash ready:" << h << ". Start blocks download...";
				transition(Asking::Blocks);
				return true;
			}
			else if (status == QueueStatus::Bad)
			{
				cwarn << "block hash bad!" << h << ". Bailing...";
				transition(Asking::Nothing);
				return true;
			}
			else if (status == QueueStatus::Unknown)
			{
				unknowns++;
				m_syncingNeededBlocks.push_back(h);
			}
			else
				knowns++;
			m_syncingLastReceivedHash = h;
		}
		clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns; now at" << m_syncingLastReceivedHash;
		// run through - ask for more.
		transition(Asking::Hashes);
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
			if (host()->m_chain.isKnown(h))
			{
				rlp += host()->m_chain.block(_r[i].toHash<h256>());
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
		unsigned itemCount = _r.itemCount();
		clog(NetMessageSummary) << "Blocks (" << dec << itemCount << "entries)" << (itemCount ? "" : ": NoMoreBlocks");

		if (m_asking != Asking::Blocks)
			clog(NetWarn) << "Unexpected Blocks received!";

		if (itemCount == 0)
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

		for (unsigned i = 0; i < itemCount; ++i)
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
				case ImportResult::BadChain:
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

				default:;
				}
			}
			else
			{
				addRating(0);	// -1?
				repeated++;
			}
		}

		clog(NetMessageSummary) << dec << success << "imported OK," << unknown << "with unknown parents," << future << "with future timestamps," << got << " already known," << repeated << " repeats received.";

		if (m_asking == Asking::Blocks)
		{
			if (!got)
				transition(Asking::Blocks);
			else
				transition(Asking::Nothing);
		}
		break;
	}
	case NewBlockPacket:
	{
		auto h = BlockInfo::headerHash(_r[0].data());
		clog(NetMessageSummary) << "NewBlock: " << h;

		if (_r.itemCount() != 2)
			disable("NewBlock without 2 data fields.");
		else
		{
			switch (host()->m_bq.import(_r[0].data(), host()->m_chain))
			{
			case ImportResult::Success:
				addRating(100);
				break;
			case ImportResult::FutureTime:
				//TODO: Rating dependent on how far in future it is.
				break;

			case ImportResult::Malformed:
			case ImportResult::BadChain:
				disable("Malformed block received.");
				return true;

			case ImportResult::AlreadyInChain:
			case ImportResult::AlreadyKnown:
				break;

			case ImportResult::UnknownParent:
				clog(NetMessageSummary) << "Received block with no known parent. Resyncing...";
				setNeedsSyncing(h, _r[1].toInt<u256>());
				break;
			default:;
			}

			DEV_GUARDED(x_knownBlocks)
				m_knownBlocks.insert(h);
		}
		break;
	}
	case NewBlockHashesPacket:
	{
		clog(NetMessageSummary) << "NewBlockHashes";
		if (host()->isSyncing())
			clog(NetMessageSummary) << "Ignoring since we're already downloading.";
		else
		{
			unsigned knowns = 0;
			unsigned unknowns = 0;
			unsigned itemCount = _r.itemCount();
			for (unsigned i = 0; i < itemCount; ++i)
			{
				addRating(1);
				auto h = _r[i].toHash<h256>();
				DEV_GUARDED(x_knownBlocks)
					m_knownBlocks.insert(h);
				auto status = host()->m_bq.blockStatus(h);
				if (status == QueueStatus::Importing || status == QueueStatus::Ready || host()->m_chain.isKnown(h))
					knowns++;
				else if (status == QueueStatus::Bad)
				{
					cwarn << "block hash bad!" << h << ". Bailing...";
					return true;
				}
				else if (status == QueueStatus::Unknown)
				{
					unknowns++;
					m_syncingNeededBlocks.push_back(h);
				}
				else
					knowns++;
			}
			clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns";
			if (unknowns > 0)
			{
				clog(NetNote) << "Not syncing and new block hash discovered: syncing without help.";
				host()->m_man.resetToChain(m_syncingNeededBlocks);
				host()->changeSyncer(this, false);
				transition(Asking::Blocks, false, false);	// TODO: transaction(Asking::NewBlocks, false)
			}
			return true;
		}
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
