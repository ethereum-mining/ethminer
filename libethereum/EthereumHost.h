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
/** @file EthereumHost.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include <utility>
#include <thread>
#include <libdevcore/Guards.h>
#include <libdevcore/Worker.h>
#include <libethcore/CommonEth.h>
#include <libp2p/Common.h>
#include "CommonNet.h"
#include "EthereumPeer.h"

namespace dev
{

class RLPStream;

namespace eth
{

class TransactionQueue;
class BlockQueue;

using UnsignedRange = std::pair<unsigned, unsigned>;
using UnsignedRanges = std::vector<UnsignedRange>;

class RangeMask
{
public:
	RangeMask() {}
	RangeMask(unsigned _begin, unsigned _end): m_ranges({{_begin, _end}}) {}

	RangeMask& operator+=(RangeMask const& _m)
	{
		for (auto const& i: _m.m_ranges)
			operator+=(i);
		return *this;
	}
	RangeMask& operator+=(UnsignedRange const& _m)
	{
		for (auto i = _m.first; i < _m.second;)
		{
			// for each number, we find the element equal or next lower. this must contain the value.
			auto it = m_ranges.lower_bound(i);
			auto uit = m_ranges.upper_bound(i + 1);
			if (it == m_ranges.end() || it->second < i)
				// lower range is too low to merge.
				// if the next higher range is too high.
				if (uit == m_ranges.end() || uit->first > _m.second)
				{
					// just create a new range
					m_ranges[i] = _m.second;
					break;
				}
				else
				{
					if (uit->first == i)
						// move i to end of range
						i = uit->second;
					else
					{
						// merge with the next higher range
						// move i to end of range
						i = m_ranges[i] = uit->second;
						i = uit->second;
						m_ranges.erase(uit);
					}
				}
			else if (it->second == i)
			{
				// if the next higher range is too high.
				if (uit == m_ranges.end() || uit->first > _m.second)
				{
					// merge with the next lower range
					m_ranges[it->first] = _m.second;
					break;
				}
				else
				{
					// merge with both next lower & next higher.
					i = m_ranges[it->first] = uit->second;
					m_ranges.erase(uit);
				}
			}
			else
				i = it->second;
		}
		return *this;
	}

	RangeMask& operator+=(unsigned _i)
	{
		return operator+=(UnsignedRange(_i, _i + 1));
	}

	bool contains(unsigned _i) const
	{
		auto it = m_ranges.lower_bound(_i);
		return it != m_ranges.end() && it->first <= _i && it->second > _i;
	}

private:
	std::map<unsigned, unsigned> m_ranges;
};

#if 0
class DownloadSub
{
	friend class DownloadMan;

public:
	h256s nextFetch();
	void noteBlock(h256 _hash, bytesConstRef _data);

private:
	void resetFetch();		// Called by DownloadMan when we need to reset the download.

	DownloadMan* m_man;

	Mutex m_fetch;
	h256s m_fetching;
	h256s m_activeGet;
	bool m_killFetch;
	RangeMask m_attempted;
};

class DownloadMan
{
	friend class DownloadSub;

public:
	void resetToChain(h256s const& _chain);

private:
	void cancelFetch(DownloadSub* );
	void noteBlock(h256 _hash, bytesConstRef _data);

	h256s m_chain;
	RangeMask m_complete;
	std::map<DownloadSub*, UnsignedRange> m_fetching;
};
#endif

/**
 * @brief The EthereumHost class
 * @warning None of this is thread-safe. You have been warned.
 */
class EthereumHost: public p2p::HostCapability<EthereumPeer>, Worker
{
	friend class EthereumPeer;

public:
	/// Start server, but don't listen.
	EthereumHost(BlockChain const& _ch, TransactionQueue& _tq, BlockQueue& _bq, u256 _networkId);

	/// Will block on network process events.
	virtual ~EthereumHost();

	unsigned protocolVersion() const { return c_protocolVersion; }
	u256 networkId() const { return m_networkId; }
	void setNetworkId(u256 _n) { m_networkId = _n; }

	void reset();

private:
	void noteHavePeerState(EthereumPeer* _who);
	/// Session wants to pass us a block that we might not have.
	/// @returns true if we didn't have it.
	bool noteBlock(h256 _hash, bytesConstRef _data);
	/// Session has finished getting the chain of hashes.
	void noteHaveChain(EthereumPeer* _who);
	/// Called when the peer can no longer provide us with any needed blocks.
	void noteDoneBlocks();

	/// Sync with the BlockChain. It might contain one of our mined blocks, we might have new candidates from the network.
	void doWork();

	/// Called by peer to add incoming transactions.
	void addIncomingTransaction(bytes const& _bytes) { std::lock_guard<std::recursive_mutex> l(m_incomingLock); m_incomingTransactions.push_back(_bytes); }

	void maintainTransactions(TransactionQueue& _tq, h256 _currentBlock);
	void maintainBlocks(BlockQueue& _bq, h256 _currentBlock);

	/// Get a bunch of needed blocks.
	/// Removes them from our list of needed blocks.
	/// @returns empty if there's no more blocks left to fetch, otherwise the blocks to fetch.
	h256Set neededBlocks(h256Set const& _exclude);

	///	Check to see if the network peer-state initialisation has happened.
	bool isInitialised() const { return m_latestBlockSent; }

	/// Initialises the network peer-state, doing the stuff that needs to be once-only. @returns true if it really was first.
	bool ensureInitialised(TransactionQueue& _tq);

	virtual void onStarting() { startWorking(); }
	virtual void onStopping() { stopWorking(); }

	void readyForSync();
	void updateGrabbing(Grabbing _g);

	BlockChain const& m_chain;
	TransactionQueue& m_tq;					///< Maintains a list of incoming transactions not yet in a block on the blockchain.
	BlockQueue& m_bq;						///< Maintains a list of incoming blocks not yet on the blockchain (to be imported).

	u256 m_networkId;

	Grabbing m_grabbing = Grabbing::Nothing;

	mutable std::recursive_mutex m_incomingLock;
	std::vector<bytes> m_incomingTransactions;
	std::vector<bytes> m_incomingBlocks;

	mutable std::mutex x_blocksNeeded;
	u256 m_totalDifficultyOfNeeded;
	h256s m_blocksNeeded;
	h256Set m_blocksOnWay;

	h256 m_latestBlockSent;
	h256Set m_transactionsSent;
};

}
}
