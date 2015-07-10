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
/** @file BlockInfo.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libdevcore/SHA3.h>
#include "Common.h"
#include "Exceptions.h"

namespace dev
{
namespace eth
{

enum IncludeProof
{
	WithoutProof = 0,
	WithProof = 1
};

enum Strictness
{
	CheckEverything,
	QuickNonce,
	IgnoreNonce,
	CheckNothing
};

/** @brief Encapsulation of a block header.
 * Class to contain all of a block header's data. It is able to parse a block header and populate
 * from some given RLP block serialisation with the static fromHeader(), through the method
 * populateFromHeader(). This will conduct a minimal level of verification. In this case extra
 * verification can be performed through verifyInternals() and verifyParent().
 *
 * The object may also be populated from an entire block through the explicit
 * constructor BlockInfo(bytesConstRef) and manually with the populate() method. These will
 * conduct verification of the header against the other information in the block.
 *
 * The object may be populated with a template given a parent BlockInfo object with the
 * populateFromParent() method. The genesis block info may be retrieved with genesis() and the
 * corresponding RLP block created with createGenesisBlock().
 *
 * The difficulty and gas-limit derivations may be calculated with the calculateDifficulty()
 * and calculateGasLimit() and the object serialised to RLP with streamRLP. To determine the
 * header hash without the nonce (for mining), the method headerHash(WithoutNonce) is provided.
 *
 * The default constructor creates an empty object, which can be tested against with the boolean
 * conversion operator.
 */
struct BlockInfo
{
public:
	// TODO: make them all private!
	h256 parentHash;
	h256 sha3Uncles;
	Address coinbaseAddress;
	h256 stateRoot;
	h256 transactionsRoot;
	h256 receiptsRoot;
	LogBloom logBloom;
	u256 difficulty;
	u256 number;
	u256 gasLimit;
	u256 gasUsed;
	u256 timestamp = Invalid256;
	bytes extraData;

	BlockInfo();
	BlockInfo(bytesConstRef _block, Strictness _s);

	static h256 headerHashFromBlock(bytes const& _block) { return headerHashFromBlock(&_block); }
	static h256 headerHashFromBlock(bytesConstRef _block);

	explicit operator bool() const { return timestamp != Invalid256; }

	bool operator==(BlockInfo const& _cmp) const
	{
		return parentHash == _cmp.parentHash &&
			sha3Uncles == _cmp.sha3Uncles &&
			coinbaseAddress == _cmp.coinbaseAddress &&
			stateRoot == _cmp.stateRoot &&
			transactionsRoot == _cmp.transactionsRoot &&
			receiptsRoot == _cmp.receiptsRoot &&
			logBloom == _cmp.logBloom &&
			difficulty == _cmp.difficulty &&
			number == _cmp.number &&
			gasLimit == _cmp.gasLimit &&
			gasUsed == _cmp.gasUsed &&
			timestamp == _cmp.timestamp &&
			extraData == _cmp.extraData;
	}
	bool operator!=(BlockInfo const& _cmp) const { return !operator==(_cmp); }

	void verifyInternals(bytesConstRef _block) const;
	void verifyParent(BlockInfo const& _parent) const;
	void populateFromParent(BlockInfo const& parent);

	u256 calculateDifficulty(BlockInfo const& _parent) const;
	u256 selectGasLimit(BlockInfo const& _parent) const;
	h256 const& boundary() const;

	/// sha3 of the header only.
	h256 const& hashWithout() const;
	h256 const& hash() const { return m_hash; }

protected:
	static RLP extractHeader(bytesConstRef _block);
	void populateFromHeader(RLP const& _header, Strictness _s = IgnoreNonce);
	void streamRLPFields(RLPStream& _s) const;

	void clear();
	void noteDirty() const { m_hashWithout = m_boundary = m_hash = h256(); }

	static const unsigned BasicFields = 13;

	mutable h256 m_hash;						///< SHA3 hash of the block header! Not serialised.

private:
	mutable h256 m_hashWithout;					///< SHA3 hash of the block header! Not serialised.
	mutable h256 m_boundary;					///< 2^256 / difficulty
};

inline std::ostream& operator<<(std::ostream& _out, BlockInfo const& _bi)
{
	_out << _bi.hashWithout() << " " << _bi.parentHash << " " << _bi.sha3Uncles << " " << _bi.coinbaseAddress << " " << _bi.stateRoot << " " << _bi.transactionsRoot << " " <<
			_bi.receiptsRoot << " " << _bi.logBloom << " " << _bi.difficulty << " " << _bi.number << " " << _bi.gasLimit << " " <<
			_bi.gasUsed << " " << _bi.timestamp;
	return _out;
}

template <class BlockInfoSub>
class BlockHeaderPolished: public BlockInfoSub
{
public:
	BlockHeaderPolished() {}
	BlockHeaderPolished(BlockInfo const& _bi): BlockInfoSub(_bi) {}
	explicit BlockHeaderPolished(bytes const& _block, Strictness _s = IgnoreNonce, h256 const& _h = h256()) { populate(&_block, _s, _h); }
	explicit BlockHeaderPolished(bytesConstRef _block, Strictness _s = IgnoreNonce, h256 const& _h = h256()) { populate(_block, _s, _h); }

	static BlockHeaderPolished fromHeader(bytes const& _header, Strictness _s = IgnoreNonce, h256 const& _h = h256()) { return fromHeader(bytesConstRef(&_header), _s, _h); }
	static BlockHeaderPolished fromHeader(bytesConstRef _header, Strictness _s = IgnoreNonce, h256 const& _h = h256()) { BlockHeaderPolished ret; ret.populateFromHeader(_header, _s, _h); return ret; }

	void populate(bytesConstRef _block, Strictness _s, h256 const& _h = h256()) { populateFromHeader(BlockInfo::extractHeader(_block), _s, _h); }

	void populateFromParent(BlockHeaderPolished const& _parent)
	{
		noteDirty();
		BlockInfo::parentHash = _parent.hash();
		BlockInfo::populateFromParent(_parent);
	}

	void verifyParent(BlockHeaderPolished const& _parent)
	{
		if (BlockInfo::parentHash && BlockInfo::parentHash != _parent.hash())
			BOOST_THROW_EXCEPTION(InvalidParentHash());
		BlockInfo::verifyParent(_parent);
	}

	void populateFromHeader(RLP const& _header, Strictness _s = IgnoreNonce, h256 const& _h = h256())
	{
		BlockInfo::m_hash = _h;
		if (_h)
			assert(_h == dev::sha3(_header.data()));

		if (_header.itemCount() != BlockInfo::BasicFields + BlockInfoSub::SealFields)
			BOOST_THROW_EXCEPTION(InvalidBlockHeaderItemCount());

		BlockInfo::populateFromHeader(_header, _s);
		BlockInfoSub::populateFromHeader(_header, _s);
	}

	void clear() { BlockInfo::clear(); BlockInfoSub::clear(); BlockInfoSub::noteDirty(); }
	void noteDirty() const { BlockInfo::noteDirty(); BlockInfoSub::noteDirty(); }

	h256 headerHash(IncludeProof _i = WithProof) const
	{
		RLPStream s;
		streamRLP(s, _i);
		return sha3(s.out());
	}

	h256 const& hash() const
	{
		if (!BlockInfo::m_hash)
			BlockInfo::m_hash = headerHash(WithProof);
		return BlockInfo::m_hash;
	}

	void streamRLP(RLPStream& _s, IncludeProof _i = WithProof) const
	{
		_s.appendList(BlockInfo::BasicFields + (_i == WithProof ? BlockInfoSub::SealFields : 0));
		BlockInfo::streamRLPFields(_s);
		if (_i == WithProof)
			BlockInfoSub::streamRLPFields(_s);
	}
};

}
}
