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
#include "Exceptions.h"

namespace dev
{
namespace eth
{

/// An Ethereum address: 20 bytes.
using Address = h160;

/// The log bloom's size (2048-bit).
using LogBloom = h2048;

using Nonce = h64;

using BlockNumber = unsigned;

enum Strictness
{
	CheckEverything,
	IgnoreSeal,
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
class BlockInfo
{
public:
	static const unsigned BasicFields = 13;

	BlockInfo();
	explicit BlockInfo(bytesConstRef _data, Strictness _s = CheckEverything, h256 const& _hashWith = h256());
	explicit BlockInfo(bytes const& _data, Strictness _s = CheckEverything, h256 const& _hashWith = h256()): BlockInfo(&_data, _s, _hashWith) {}

	static RLP extractHeader(bytesConstRef _block);

	explicit operator bool() const { return m_timestamp != Invalid256; }

	bool operator==(BlockInfo const& _cmp) const
	{
		return m_parentHash == _cmp.parentHash() &&
			m_sha3Uncles == _cmp.sha3Uncles() &&
			m_coinbaseAddress == _cmp.beneficiary() &&
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

	h256 const& boundary() const;

	h256 const& parentHash() const { return m_parentHash; }
	h256 const& sha3Uncles() const { return m_sha3Uncles; }

	void setNumber(u256 const& _v) { m_number = _v; noteDirty(); }
	void setDifficulty(u256 const& _v) { m_difficulty = _v; noteDirty(); }

	Address const& beneficiary() const { return m_coinbaseAddress; }
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

template <class BlockInfoSub>
class BlockHeaderPolished: public BlockInfoSub
{
public:
	BlockHeaderPolished() {}
	BlockHeaderPolished(BlockInfo const& _bi): BlockInfoSub(_bi) {}
	explicit BlockHeaderPolished(bytes const& _data, Strictness _s = IgnoreSeal, h256 const& _h = h256()) { populate(&_data, _s, _h); }
	explicit BlockHeaderPolished(bytesConstRef _data, Strictness _s = IgnoreSeal, h256 const& _h = h256()) { populate(_data, _s, _h); }

	// deprecated for public API - use constructor.
	// TODO: make private.
	void populate(bytesConstRef _data, Strictness _s, h256 const& _h = h256())
	{
		populateFromHeader(BlockInfo::extractHeader(_data), _s, _h);
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

	h256 headerHash() const
	{
		RLPStream s;
		streamRLP(s);
		return sha3(s.out());
	}

	h256 const& hash() const
	{
		if (!BlockInfo::m_hash)
			BlockInfo::m_hash = headerHash();
		return BlockInfo::m_hash;
	}

	void streamRLP(RLPStream& _s) const
	{
		_s.appendList(BlockInfo::BasicFields + BlockInfoSub::SealFields);
		BlockInfo::streamRLPFields(_s);
		BlockInfoSub::streamRLPFields(_s);
	}
};

class BlockHeaderRaw: public BlockInfo
{
public:
	h256 const& seedHash() const;
	Nonce const& nonce() const { return m_nonce; }

protected:
	BlockHeaderRaw() = default;
	BlockHeaderRaw(BlockInfo const& _bi): BlockInfo(_bi) {}

	void clear() { m_mixHash = h256(); m_nonce = Nonce(); }

private:
	Nonce m_nonce;
	h256 m_mixHash;

	mutable h256 m_seedHash;
};

using BlockHeader = BlockHeaderPolished<BlockHeaderRaw>;

}
}
