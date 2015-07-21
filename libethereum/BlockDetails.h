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
/** @file BlockDetails.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <unordered_map>
#include <libdevcore/db.h>
#include <libdevcore/Log.h>
#include <libdevcore/RLP.h>
#include "TransactionReceipt.h"

namespace dev
{
namespace eth
{

// TODO: OPTIMISE: constructors take bytes, RLP used only in necessary classes.

static const unsigned c_bloomIndexSize = 16;
static const unsigned c_bloomIndexLevels = 2;

struct BlockDetails
{
	BlockDetails(): number(0), totalDifficulty(0) {}
	BlockDetails(unsigned _n, u256 _tD, h256 _p, h256s _c): number(_n), totalDifficulty(_tD), parent(_p), children(_c) {}
	BlockDetails(RLP const& _r);
	bytes rlp() const;

	bool isNull() const { return !totalDifficulty; }
	explicit operator bool() const { return !isNull(); }

	unsigned number;
	u256 totalDifficulty;
	h256 parent;
	h256s children;

	mutable unsigned size;
};

struct BlockLogBlooms
{
	BlockLogBlooms() {}
	BlockLogBlooms(RLP const& _r) { blooms = _r.toVector<LogBloom>(); size = _r.data().size(); }
	bytes rlp() const { bytes r = dev::rlp(blooms); size = r.size(); return r; }

	LogBlooms blooms;
	mutable unsigned size;
};

struct BlocksBlooms
{
	BlocksBlooms() {}
	BlocksBlooms(RLP const& _r) { blooms = _r.toArray<LogBloom, c_bloomIndexSize>(); size = _r.data().size(); }
	bytes rlp() const { bytes r = dev::rlp(blooms); size = r.size(); return r; }

	std::array<LogBloom, c_bloomIndexSize> blooms;
	mutable unsigned size;
};

struct BlockReceipts
{
	BlockReceipts() {}
	BlockReceipts(RLP const& _r) { for (auto const& i: _r) receipts.emplace_back(i.data()); size = _r.data().size(); }
	bytes rlp() const { RLPStream s(receipts.size()); for (TransactionReceipt const& i: receipts) i.streamRLP(s); size = s.out().size(); return s.out(); }

	TransactionReceipts receipts;
	mutable unsigned size;
};

struct BlockHash
{
	BlockHash() {}
	BlockHash(h256 const& _h): value(_h) {}
	BlockHash(RLP const& _r) { value = _r.toHash<h256>(); }
	bytes rlp() const { return dev::rlp(value); }

	h256 value;
	static const unsigned size = 65;
};

struct TransactionAddress
{
	TransactionAddress() {}
	TransactionAddress(RLP const& _rlp) { blockHash = _rlp[0].toHash<h256>(); index = _rlp[1].toInt<unsigned>(); }
	bytes rlp() const { RLPStream s(2); s << blockHash << index; return s.out(); }

	explicit operator bool() const { return !!blockHash; }

	h256 blockHash;
	unsigned index = 0;

	static const unsigned size = 67;
};

using BlockDetailsHash = std::unordered_map<h256, BlockDetails>;
using BlockLogBloomsHash = std::unordered_map<h256, BlockLogBlooms>;
using BlockReceiptsHash = std::unordered_map<h256, BlockReceipts>;
using TransactionAddressHash = std::unordered_map<h256, TransactionAddress>;
using BlockHashHash = std::unordered_map<uint64_t, BlockHash>;
using BlocksBloomsHash = std::unordered_map<h256, BlocksBlooms>;

static const BlockDetails NullBlockDetails;
static const BlockLogBlooms NullBlockLogBlooms;
static const BlockReceipts NullBlockReceipts;
static const TransactionAddress NullTransactionAddress;
static const BlockHash NullBlockHash;
static const BlocksBlooms NullBlocksBlooms;

}
}
