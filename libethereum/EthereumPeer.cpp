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
	m_sub(host()->m_man),
	m_hashSub(host()->m_hashMan)
{
	requestState();
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
	if (m_asking == Asking::Blocks)
	{
		clog(NetNote) << "Finishing blocks fetch...";
		// NOTE: need to notify of giving up on chain-hashes, too, altering state as necessary.
		m_sub.doneFetch();
		m_hashSub.doneFetch();

		setAsking(Asking::Nothing);
	}
	else if (m_asking == Asking::Hashes)
	{
		clog(NetNote) << "Finishing hashes fetch...";

		setAsking(Asking::Nothing);
	}
	else if (m_asking == Asking::State)
	{
		setAsking(Asking::Nothing);
	}
}

void EthereumPeer::requestState()
{
	if (m_asking != Asking::Nothing)
	clog(NetWarn) << "Bad state: requesting state should be the first action";
	setAsking(Asking::State);
	RLPStream s;
	prep(s, StatusPacket, 5)
					<< host()->protocolVersion() - 1
					<< host()->networkId()
					<< host()->m_chain.details().totalDifficulty
					<< host()->m_chain.currentHash()
					<< host()->m_chain.genesisHash();
	sealAndSend(s);
}

void EthereumPeer::requestHashes()
{
	assert(m_asking != Asking::Blocks);
	m_syncHashNumber = m_hashSub.nextFetch(c_maxBlocksAsk);
	setAsking(Asking::Hashes);
	RLPStream s;
	prep(s, GetBlockHashesPacket, 2) << m_syncHashNumber << c_maxHashesAsk;
	sealAndSend(s);
}

void EthereumPeer::requestHashes(h256 const& _lastHash)
{
	assert(m_asking != Asking::Blocks);
	setAsking(Asking::Hashes);
	RLPStream s;
	prep(s, GetBlockHashesPacket, 2) << _lastHash << c_maxHashesAsk;
	sealAndSend(s);
}

void EthereumPeer::requestBlocks()
{
	// Looks like it's the best yet for total difficulty. Set to download.
	setAsking(Asking::Blocks);		// will kick off other peers to help if available.
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

	if (!isSyncing())
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
		if (!!session()->cap<EthereumPeer>(EthereumHost::staticVersion()))
			m_protocolVersion = host()->protocolVersion();
		m_networkId = _r[1].toInt<u256>();

		// a bit dirty as we're misusing these to communicate the values to transition, but harmless.
		m_totalDifficulty = _r[2].toInt<u256>();
		m_latestHash = _r[3].toHash<h256>();
		m_genesisHash = _r[4].toHash<h256>();
		clog(NetMessageSummary) << "Status:" << m_protocolVersion << "/" << m_networkId << "/" << m_genesisHash << ", TD:" << m_totalDifficulty << "=" << m_latestHash;
		host()->onPeerState(this);
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
		if (m_protocolVersion == host()->protocolVersion())
		{
			u256 number256 = _r[0].toInt<u256>();
			unsigned number = (unsigned) number256;
			unsigned limit = _r[1].toInt<unsigned>();
			clog(NetMessageSummary) << "GetBlockHashes (" << number << "-" << number + limit << ")";
			RLPStream s;
			if (number <= host()->m_chain.number())
			{
				unsigned c = min<unsigned>(host()->m_chain.number() - number + 1, limit);
				prep(s, BlockHashesPacket, c);
				for (unsigned n = number; n < number + c; n++)
				{
					h256 p = host()->m_chain.numberHash(n);
					s << p;
				}
			}
			else
				prep(s, BlockHashesPacket, 0);
			sealAndSend(s);
			addRating(0);
		}
		else
		{
			// Support V60 protocol
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
		}
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
			host()->onPeerDoneHashes(this, false);
			return true;
		}
		h256s hashes(itemCount);
		for (unsigned i = 0; i < itemCount; ++i)
		{
			hashes[i] = _r[i].toHash<h256>();
			m_hashSub.noteHash(m_syncHashNumber + i, 1);
		}

		if (m_protocolVersion == host()->protocolVersion())
		{
			//v61, report hashes ordered by number
			host()->onPeerHashes(this,  m_syncHashNumber, hashes);
		}
		else
			host()->onPeerHashes(this, hashes);
		m_syncHashNumber += itemCount;
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
		host()->onPeerBlocks(this, _r);
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
			unsigned itemCount = _r.itemCount();
			clog(NetMessageSummary) << "BlockHashes (" << dec << itemCount << "entries)" << (itemCount ? "" : ": NoMoreHashes");

			h256s hashes(itemCount);
			for (unsigned i = 0; i < itemCount; ++i)
				hashes[i] = _r[i].toHash<h256>();

			clog(NetNote) << "Not syncing and new block hash discovered: syncing without help.";
			host()->onPeerHashes(this, hashes);
			host()->onPeerDoneHashes(this, true);
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
