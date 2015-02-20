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

#pragma warning(push)
#pragma warning(disable: 4100 4267)
#include <leveldb/db.h>
#pragma warning(pop)

#include <libdevcore/Log.h>
#include <libdevcore/RLP.h>
#include "TransactionReceipt.h"
namespace ldb = leveldb;

namespace dev
{
namespace eth
{

struct BlockDetails
{
	BlockDetails(): number(0), totalDifficulty(0) {}
	BlockDetails(unsigned _n, u256 _tD, h256 _p, h256s _c): number(_n), totalDifficulty(_tD), parent(_p), children(_c) {}
	BlockDetails(RLP const& _r);
	bytes rlp() const;

	bool isNull() const { return !totalDifficulty; }
	explicit operator bool() const { return !isNull(); }

	unsigned number;			// TODO: remove?
	u256 totalDifficulty;
	h256 parent;
	h256s children;
};

struct BlockLogBlooms
{
	BlockLogBlooms() {}
	BlockLogBlooms(RLP const& _r) { blooms = _r.toVector<h512>(); }
	bytes rlp() const { RLPStream s; s << blooms; return s.out(); }

	h512s blooms;
};

struct BlockReceipts
{
	BlockReceipts() {}
	BlockReceipts(RLP const& _r) { for (auto const& i: _r) receipts.emplace_back(i.data()); }
	bytes rlp() const { RLPStream s(receipts.size()); for (TransactionReceipt const& i: receipts) i.streamRLP(s); return s.out(); }

	TransactionReceipts receipts;
};

using BlockDetailsHash = std::map<h256, BlockDetails>;
using BlockLogBloomsHash = std::map<h256, BlockLogBlooms>;
using BlockReceiptsHash = std::map<h256, BlockReceipts>;

static const BlockDetails NullBlockDetails;
static const BlockLogBlooms NullBlockLogBlooms;
static const BlockReceipts NullBlockReceipts;

}
}
