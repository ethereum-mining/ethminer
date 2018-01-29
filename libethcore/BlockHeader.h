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

using BlockNumber = unsigned;


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
class BlockHeader
{
public:
	static const unsigned BasicFields = 13;

	BlockHeader() = default;
	explicit BlockHeader(bytesConstRef _data);
	explicit BlockHeader(bytes const& _data): BlockHeader(&_data) {}

	static RLP extractHeader(bytesConstRef _block);

	explicit operator bool() const { return m_timestamp != Invalid256; }

	h256 const& boundary() const;

	void setNumber(u256 const& _v) { m_number = _v; noteDirty(); }
	void setDifficulty(u256 const& _v) { m_difficulty = _v; noteDirty(); }

	u256 const& number() const { return m_number; }

	/// sha3 of the header only.
	h256 const& hashWithout() const;

	void noteDirty() const { m_hashWithout = m_boundary = h256(); }

	uint64_t nonce() const { return m_nonce; }

private:
	void populateFromHeader(RLP const& _header);
	void streamRLPFields(RLPStream& _s) const;

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

	u256 m_difficulty;

	mutable h256 m_hashWithout;					///< SHA3 hash of the block header! Not serialised.
	mutable h256 m_boundary;					///< 2^256 / difficulty

	uint64_t m_nonce;
	mutable h256 m_seedHash;
};

}
}
