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
	JustSeal,
	QuickNonce,
	IgnoreSeal,
	CheckNothing
};

enum BlockDataType
{
	HeaderData,
	BlockData
};

DEV_SIMPLE_EXCEPTION(NoHashRecorded);
DEV_SIMPLE_EXCEPTION(GenesisBlockCannotBeCalculated);

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
class BlockInfo
{
	friend class BlockChain;
public:
	static const unsigned BasicFields = 13;

	BlockInfo();
	explicit BlockInfo(bytesConstRef _data, Strictness _s = CheckEverything, h256 const& _hashWith = h256(), BlockDataType _bdt = BlockData);
	explicit BlockInfo(bytes const& _data, Strictness _s = CheckEverything, h256 const& _hashWith = h256(), BlockDataType _bdt = BlockData): BlockInfo(&_data, _s, _hashWith, _bdt) {}

	static h256 headerHashFromBlock(bytes const& _block) { return headerHashFromBlock(&_block); }
	static h256 headerHashFromBlock(bytesConstRef _block);
	static RLP extractHeader(bytesConstRef _block);

	explicit operator bool() const { return m_timestamp != Invalid256; }

	bool operator==(BlockInfo const& _cmp) const
	{
		return m_parentHash == _cmp.parentHash() &&
			m_sha3Uncles == _cmp.sha3Uncles() &&
			m_coinbaseAddress == _cmp.coinbaseAddress() &&
			m_stateRoot == _cmp.stateRoot() &&
			m_transactionsRoot == _cmp.transactionsRoot() &&
			m_receiptsRoot == _cmp.receiptsRoot() &&
			m_logBloom == _cmp.logBloom() &&
			m_difficulty == _cmp.difficulty() &&
			m_number == _cmp.number() &&
			m_gasLimit == _cmp.gasLimit() &&
			m_gasUsed == _cmp.gasUsed() &&
			m_timestamp == _cmp.timestamp() &&
			m_extraData == _cmp.extraData();
	}
	bool operator!=(BlockInfo const& _cmp) const { return !operator==(_cmp); }

	void verifyInternals(bytesConstRef _block) const;
	void verifyParent(BlockInfo const& _parent) const;
	void populateFromParent(BlockInfo const& parent);

	u256 calculateDifficulty(BlockInfo const& _parent) const;
	u256 childGasLimit(u256 const& _gasFloorTarget = UndefinedU256) const;
	h256 const& boundary() const;

	h256 const& parentHash() const { return m_parentHash; }
	h256 const& sha3Uncles() const { return m_sha3Uncles; }

	void setParentHash(h256 const& _v) { m_parentHash = _v; noteDirty(); }
	void setSha3Uncles(h256 const& _v) { m_sha3Uncles = _v; noteDirty(); }
	void setTimestamp(u256 const& _v) { m_timestamp = _v; noteDirty(); }
	void setCoinbaseAddress(Address const& _v) { m_coinbaseAddress = _v; noteDirty(); }
	void setRoots(h256 const& _t, h256 const& _r, h256 const& _u, h256 const& _s) { m_transactionsRoot = _t; m_receiptsRoot = _r; m_stateRoot = _s; m_sha3Uncles = _u; noteDirty(); }
	void setGasUsed(u256 const& _v) { m_gasUsed = _v; noteDirty(); }
	void setNumber(u256 const& _v) { m_number = _v; noteDirty(); }
	void setGasLimit(u256 const& _v) { m_gasLimit = _v; noteDirty(); }
	void setExtraData(bytes const& _v) { m_extraData = _v; noteDirty(); }
	void setLogBloom(LogBloom const& _v) { m_logBloom = _v; noteDirty(); }
	void setDifficulty(u256 const& _v) { m_difficulty = _v; noteDirty(); }

	Address const& coinbaseAddress() const { return m_coinbaseAddress; }
	h256 const& stateRoot() const { return m_stateRoot; }
	h256 const& transactionsRoot() const { return m_transactionsRoot; }
	h256 const& receiptsRoot() const { return m_receiptsRoot; }
	LogBloom const& logBloom() const { return m_logBloom; }
	u256 const& number() const { return m_number; }
	u256 const& gasLimit() const { return m_gasLimit; }
	u256 const& gasUsed() const { return m_gasUsed; }
	u256 const& timestamp() const { return m_timestamp; }
	bytes const& extraData() const { return m_extraData; }

	u256 const& difficulty() const { return m_difficulty; }		// TODO: pull out into BlockHeader

	/// sha3 of the header only.
	h256 const& hashWithout() const;
	h256 const& hash() const { if (m_hash) return m_hash; BOOST_THROW_EXCEPTION(NoHashRecorded()); }

	void clear();
	void noteDirty() const { m_hashWithout = m_boundary = m_hash = h256(); }

protected:
	void populateFromHeader(RLP const& _header, Strictness _s = IgnoreSeal);
	void streamRLPFields(RLPStream& _s) const;

	mutable h256 m_hash;						///< SHA3 hash of the block header! Not serialised.

	h256 m_parentHash;
	h256 m_sha3Uncles;
	Address m_coinbaseAddress;
	h256 m_stateRoot;
	h256 m_transactionsRoot;
	h256 m_receiptsRoot;
	LogBloom m_logBloom;
	u256 m_number;
	u256 m_gasLimit;
	u256 m_gasUsed;
	u256 m_timestamp = Invalid256;
	bytes m_extraData;

	u256 m_difficulty;		// TODO: pull out into BlockHeader

private:
	mutable h256 m_hashWithout;					///< SHA3 hash of the block header! Not serialised.
	mutable h256 m_boundary;					///< 2^256 / difficulty
};

inline std::ostream& operator<<(std::ostream& _out, BlockInfo const& _bi)
{
	_out << _bi.hashWithout() << " " << _bi.parentHash() << " " << _bi.sha3Uncles() << " " << _bi.coinbaseAddress() << " " << _bi.stateRoot() << " " << _bi.transactionsRoot() << " " <<
			_bi.receiptsRoot() << " " << _bi.logBloom() << " " << _bi.difficulty() << " " << _bi.number() << " " << _bi.gasLimit() << " " <<
			_bi.gasUsed() << " " << _bi.timestamp();
	return _out;
}

template <class BlockInfoSub>
class BlockHeaderPolished: public BlockInfoSub
{
public:
	static const unsigned Fields = BlockInfoSub::BasicFields + BlockInfoSub::SealFields;

	BlockHeaderPolished() {}
	BlockHeaderPolished(BlockInfo const& _bi): BlockInfoSub(_bi) {}
	explicit BlockHeaderPolished(bytes const& _data, Strictness _s = IgnoreSeal, h256 const& _h = h256(), BlockDataType _bdt = BlockData) { populate(&_data, _s, _h, _bdt); }
	explicit BlockHeaderPolished(bytesConstRef _data, Strictness _s = IgnoreSeal, h256 const& _h = h256(), BlockDataType _bdt = BlockData) { populate(_data, _s, _h, _bdt); }

	// deprecated - just use constructor instead.
	static BlockHeaderPolished fromHeader(bytes const& _data, Strictness _s = IgnoreSeal, h256 const& _h = h256()) { return BlockHeaderPolished(_data, _s, _h, HeaderData); }
	static BlockHeaderPolished fromHeader(bytesConstRef _data, Strictness _s = IgnoreSeal, h256 const& _h = h256()) { return BlockHeaderPolished(_data, _s, _h, HeaderData); }

	// deprecated for public API - use constructor.
	// TODO: make private.
	void populate(bytesConstRef _data, Strictness _s, h256 const& _h = h256(), BlockDataType _bdt = BlockData)
	{
		populateFromHeader(_bdt == BlockData ? BlockInfo::extractHeader(_data) : RLP(_data), _s, _h);
	}

	void populateFromParent(BlockHeaderPolished const& _parent)
	{
		noteDirty();
		BlockInfo::m_parentHash = _parent.hash();
		BlockInfo::populateFromParent(_parent);
		BlockInfoSub::populateFromParent(_parent);
	}

	// TODO: consider making private.
	void verifyParent(BlockHeaderPolished const& _parent)
	{
		if (BlockInfo::parentHash() && BlockInfo::parentHash() != _parent.hash())
			BOOST_THROW_EXCEPTION(InvalidParentHash());
		BlockInfo::verifyParent(_parent);
		BlockInfoSub::verifyParent(_parent);
	}

	// deprecated for public API - use constructor.
	// TODO: make private.
	void populateFromHeader(RLP const& _header, Strictness _s = IgnoreSeal, h256 const& _h = h256())
	{
		BlockInfo::m_hash = _h;
		if (_h)
			assert(_h == dev::sha3(_header.data()));
		else
			BlockInfo::m_hash = dev::sha3(_header.data());

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

	bytes sealFieldsRLP() const
	{
		RLPStream s;
		BlockInfoSub::streamRLPFields(s);
		return s.out();
	}
};

}
}
