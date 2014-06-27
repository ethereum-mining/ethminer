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

#include <libethential/Common.h>
#include <libethential/RLP.h>
#include "CommonEth.h"

namespace eth
{

extern u256 c_genesisDifficulty;

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
 * and calculateGasLimit() and the object serialised to RLP with fillStream. To determine the
 * header hash without the nonce (for mining), the method headerHashWithoutNonce() is provided.
 *
 * The defualt constructor creates an empty object, which can be tested against with the boolean
 * conversion operator.
 */
struct BlockInfo
{
public:
	h256 hash;						///< SHA3 hash of the entire block! Not serialised (the only member not contained in a block header).
	h256 parentHash;
	h256 sha3Uncles;
	Address coinbaseAddress;
	h256 stateRoot;
	h256 transactionsRoot;
	u256 difficulty;
	u256 number;
	u256 minGasPrice;
	u256 gasLimit;
	u256 gasUsed;
	u256 timestamp;
	bytes extraData;
	h256 nonce;

	BlockInfo();
	explicit BlockInfo(bytesConstRef _block);

	static BlockInfo fromHeader(bytesConstRef _block);

	explicit operator bool() const { return timestamp != Invalid256; }

	bool operator==(BlockInfo const& _cmp) const
	{
		return parentHash == _cmp.parentHash &&
				sha3Uncles == _cmp.sha3Uncles &&
				coinbaseAddress == _cmp.coinbaseAddress &&
				stateRoot == _cmp.stateRoot &&
				transactionsRoot == _cmp.transactionsRoot &&
				difficulty == _cmp.difficulty &&
				number == _cmp.number &&
				minGasPrice == _cmp.minGasPrice &&
				gasLimit == _cmp.gasLimit &&
				gasUsed == _cmp.gasUsed &&
				timestamp == _cmp.timestamp &&
				extraData == _cmp.extraData &&
				nonce == _cmp.nonce;
	}
	bool operator!=(BlockInfo const& _cmp) const { return !operator==(_cmp); }

	void populateFromHeader(RLP const& _header, bool _checkNonce = true);
	void populate(bytesConstRef _block, bool _checkNonce = true);
	void verifyInternals(bytesConstRef _block) const;
	void verifyParent(BlockInfo const& _parent) const;
	void populateFromParent(BlockInfo const& parent);

	u256 calculateDifficulty(BlockInfo const& _parent) const;
	u256 calculateGasLimit(BlockInfo const& _parent) const;

	/// No-nonce sha3 of the header only.
	h256 headerHashWithoutNonce() const;
	void fillStream(RLPStream& _s, bool _nonce) const;
};

inline std::ostream& operator<<(std::ostream& _out, BlockInfo const& _bi)
{
	_out << _bi.hash << " " << _bi.parentHash << " " << _bi.sha3Uncles << " " << _bi.coinbaseAddress << " " << _bi.stateRoot << " " << _bi.transactionsRoot << " " <<
			_bi.difficulty << " " << _bi.number << " " << _bi.minGasPrice << " " << _bi.gasLimit << " " << _bi.gasUsed << " " << _bi.timestamp << " " << _bi.nonce;
	return _out;
}

}


