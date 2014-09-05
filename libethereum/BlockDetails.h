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

#include <libethential/Log.h>
#include <libethential/RLP.h>
#include "Manifest.h"
namespace ldb = leveldb;

namespace dev
{
namespace eth
{

struct BlockDetails
{
	BlockDetails(): number(0), totalDifficulty(0) {}
	BlockDetails(unsigned _n, u256 _tD, h256 _p, h256s _c, h256 _bloom): number(_n), totalDifficulty(_tD), parent(_p), children(_c), bloom(_bloom) {}
	BlockDetails(RLP const& _r);
	bytes rlp() const;

	bool isNull() const { return !totalDifficulty; }
	explicit operator bool() const { return !isNull(); }

	unsigned number;			// TODO: remove?
	u256 totalDifficulty;
	h256 parent;
	h256s children;
	h256 bloom;
};

struct BlockBlooms
{
	BlockBlooms() {}
	BlockBlooms(RLP const& _r) { blooms = _r.toVector<h256>(); }
	bytes rlp() const { RLPStream s; s << blooms; return s.out(); }

	h256s blooms;
};

struct BlockTraces
{
	BlockTraces() {}
	BlockTraces(RLP const& _r) { for (auto const& i: _r) traces.emplace_back(i.data()); }
	bytes rlp() const { RLPStream s(traces.size()); for (auto const& i: traces) i.streamOut(s); return s.out(); }

	Manifests traces;
};


typedef std::map<h256, BlockDetails> BlockDetailsHash;
typedef std::map<h256, BlockBlooms> BlockBloomsHash;
typedef std::map<h256, BlockTraces> BlockTracesHash;

static const BlockDetails NullBlockDetails;
static const BlockBlooms NullBlockBlooms;
static const BlockTraces NullBlockTraces;

}
}
