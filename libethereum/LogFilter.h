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
/** @file LogFilter.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libethcore/Common.h>
#include "TransactionReceipt.h"

namespace dev
{

namespace eth
{
class LogFilter;
}

namespace eth
{

/// Simple stream output for the StateDiff.
std::ostream& operator<<(std::ostream& _out, dev::eth::LogFilter const& _s);

class State;

class LogFilter
{
public:
	LogFilter(h256 _earliest = EarliestBlockHash, h256 _latest = PendingBlockHash): m_earliest(_earliest), m_latest(_latest) {}

	void streamRLP(RLPStream& _s) const;
	h256 sha3() const;

	/// hash of earliest block which should be filtered
	h256 earliest() const { return m_earliest; }

	/// hash of latest block which should be filtered
	h256 latest() const { return m_latest; }

	/// Range filter is a filter which doesn't care about addresses or topics
	/// Matches are all entries from earliest to latest
	/// @returns true if addresses and topics are unspecified
	bool isRangeFilter() const;

	/// @returns bloom possibilities for all addresses and topics
	std::vector<LogBloom> bloomPossibilities() const;

	bool matches(LogBloom _bloom) const;
	bool matches(State const& _s, unsigned _i) const;
	LogEntries matches(TransactionReceipt const& _r) const;

	LogFilter address(Address _a) { m_addresses.insert(_a); return *this; }
	LogFilter topic(unsigned _index, h256 const& _t) { if (_index < 4) m_topics[_index].insert(_t); return *this; }
	LogFilter withEarliest(h256 _e) { m_earliest = _e; return *this; }
	LogFilter withLatest(h256 _e) { m_latest = _e; return *this; }

	friend std::ostream& dev::eth::operator<<(std::ostream& _out, dev::eth::LogFilter const& _s);

private:
	AddressHash m_addresses;
	std::array<h256Hash, 4> m_topics;
	h256 m_earliest = EarliestBlockHash;
	h256 m_latest = PendingBlockHash;
};

}

}
