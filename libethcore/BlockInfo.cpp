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
/** @file BlockInfo.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libdevcrypto/TrieDB.h>
#include <libethcore/Common.h>
#include "ProofOfWork.h"
#include "Exceptions.h"
#include "Params.h"
#include "BlockInfo.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

BlockInfo::BlockInfo(): timestamp(Invalid256)
{
}

BlockInfo::BlockInfo(bytesConstRef _block, bool _checkNonce)
{
	populate(_block, _checkNonce);
}

void BlockInfo::setEmpty()
{
	parentHash = h256();
	sha3Uncles = EmptyListSHA3;
	coinbaseAddress = Address();
	stateRoot = EmptyTrie;
	transactionsRoot = EmptyTrie;
	receiptsRoot = EmptyTrie;
	logBloom = LogBloom();
	difficulty = 0;
	number = 0;
	gasLimit = 0;
	gasUsed = 0;
	timestamp = 0;
	extraData.clear();
	seedHash = h256();
	mixHash = h256();
	nonce = Nonce();
	hash = headerHash(WithNonce);
}

BlockInfo BlockInfo::fromHeader(bytesConstRef _block)
{
	BlockInfo ret;
	ret.populateFromHeader(RLP(_block));
	return ret;
}

h256 BlockInfo::headerHash(IncludeNonce _n) const
{
	RLPStream s;
	streamRLP(s, _n);
	return sha3(s.out());
}

void BlockInfo::streamRLP(RLPStream& _s, IncludeNonce _n) const
{
	_s.appendList(_n == WithNonce ? 16 : 14)
		<< parentHash << sha3Uncles << coinbaseAddress << stateRoot << transactionsRoot << receiptsRoot << logBloom
		<< difficulty << number << gasLimit << gasUsed << timestamp << extraData << seedHash;
	if (_n == WithNonce)
		_s << mixHash << nonce;
}

h256 BlockInfo::headerHash(bytesConstRef _block)
{
	return sha3(RLP(_block)[0].data());
}

void BlockInfo::populateFromHeader(RLP const& _header, bool _checkNonce)
{
	hash = dev::sha3(_header.data());

	int field = 0;
	try
	{
		parentHash = _header[field = 0].toHash<h256>(RLP::VeryStrict);
		sha3Uncles = _header[field = 1].toHash<h256>(RLP::VeryStrict);
		coinbaseAddress = _header[field = 2].toHash<Address>(RLP::VeryStrict);
		stateRoot = _header[field = 3].toHash<h256>(RLP::VeryStrict);
		transactionsRoot = _header[field = 4].toHash<h256>(RLP::VeryStrict);
		receiptsRoot = _header[field = 5].toHash<h256>(RLP::VeryStrict);
		logBloom = _header[field = 6].toHash<LogBloom>(RLP::VeryStrict);
		difficulty = _header[field = 7].toInt<u256>();
		number = _header[field = 8].toInt<u256>();
		gasLimit = _header[field = 9].toInt<u256>();
		gasUsed = _header[field = 10].toInt<u256>();
		timestamp = _header[field = 11].toInt<u256>();
		extraData = _header[field = 12].toBytes();
		seedHash = _header[field = 13].toHash<h256>(RLP::VeryStrict);
		mixHash = _header[field = 14].toHash<h256>(RLP::VeryStrict);
		nonce = _header[field = 15].toHash<Nonce>(RLP::VeryStrict);
	}

	catch (Exception const& _e)
	{
		_e << errinfo_name("invalid block header format") << BadFieldError(field, toHex(_header[field].data().toBytes()));
		throw;
	}

	// check it hashes according to proof of work or that it's the genesis block.
	if (_checkNonce && parentHash && !ProofOfWork::verify(*this))
		BOOST_THROW_EXCEPTION(InvalidBlockNonce(headerHash(WithoutNonce), nonce, difficulty));

	if (gasUsed > gasLimit)
		BOOST_THROW_EXCEPTION(TooMuchGasUsed() << RequirementError(bigint(gasLimit), bigint(gasUsed)) );

	if (number && extraData.size() > c_maximumExtraDataSize)
		BOOST_THROW_EXCEPTION(ExtraDataTooBig());
}

void BlockInfo::populate(bytesConstRef _block, bool _checkNonce)
{
	RLP root(_block);
	RLP header = root[0];

	if (!header.isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat(0, header.data()) << errinfo_comment("block header needs to be a list"));
	populateFromHeader(header, _checkNonce);

	if (!root[1].isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat(1, root[1].data()));
	if (!root[2].isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat(2, root[2].data()));
}

void BlockInfo::verifyInternals(bytesConstRef _block) const
{
	RLP root(_block);

	u256 mgp = (u256)-1;

	OverlayDB db;
	GenericTrieDB<OverlayDB> t(&db);
	t.init();
	unsigned i = 0;
	for (auto const& tr: root[1])
	{
		bytes k = rlp(i);
		t.insert(&k, tr.data());
		u256 gasprice = tr[1].toInt<u256>();
		mgp = min(mgp, gasprice); // the minimum gas price is not used for anything //TODO delete?
		++i;
	}
	if (transactionsRoot != t.root())
		BOOST_THROW_EXCEPTION(InvalidTransactionsHash(t.root(), transactionsRoot));

	if (sha3Uncles != sha3(root[2].data()))
		BOOST_THROW_EXCEPTION(InvalidUnclesHash());
}

void BlockInfo::populateFromParent(BlockInfo const& _parent)
{
	stateRoot = _parent.stateRoot;
	parentHash = _parent.hash;
	number = _parent.number + 1;
	gasLimit = calculateGasLimit(_parent);
	gasUsed = 0;
	difficulty = calculateDifficulty(_parent);
	seedHash = calculateSeedHash(_parent);
}

h256 BlockInfo::calculateSeedHash(BlockInfo const& _parent) const
{
	return number % c_epochDuration == 0 ? sha3(_parent.seedHash.asBytes()) : _parent.seedHash;
}

u256 BlockInfo::calculateGasLimit(BlockInfo const& _parent) const
{
	if (!parentHash)
		return c_genesisGasLimit;
	else
		return max<u256>(c_minGasLimit, (_parent.gasLimit * (c_gasLimitBoundDivisor - 1) + (_parent.gasUsed * 6 / 5)) / c_gasLimitBoundDivisor);
}

u256 BlockInfo::calculateDifficulty(BlockInfo const& _parent) const
{
	if (!parentHash)
		return (u256)c_genesisDifficulty;
	else
		return max<u256>(c_minimumDifficulty, timestamp >= _parent.timestamp + c_durationLimit ? _parent.difficulty - (_parent.difficulty / c_difficultyBoundDivisor) : (_parent.difficulty + (_parent.difficulty / c_difficultyBoundDivisor)));
}

void BlockInfo::verifyParent(BlockInfo const& _parent) const
{
	// Check difficulty is correct given the two timestamps.
	if (difficulty != calculateDifficulty(_parent))
		BOOST_THROW_EXCEPTION(InvalidDifficulty() << RequirementError((bigint)calculateDifficulty(_parent), (bigint)difficulty));

	if (gasLimit < _parent.gasLimit * (c_gasLimitBoundDivisor - 1) / c_gasLimitBoundDivisor ||
		gasLimit > _parent.gasLimit * (c_gasLimitBoundDivisor + 1) / c_gasLimitBoundDivisor)
		BOOST_THROW_EXCEPTION(InvalidGasLimit(gasLimit, _parent.gasLimit * (c_gasLimitBoundDivisor - 1) / c_gasLimitBoundDivisor, _parent.gasLimit * (c_gasLimitBoundDivisor + 1) / c_gasLimitBoundDivisor));

	if (gasLimit < _parent.gasLimit * (c_gasLimitBoundDivisor - 1) / c_gasLimitBoundDivisor ||
		gasLimit > _parent.gasLimit * (c_gasLimitBoundDivisor + 1) / c_gasLimitBoundDivisor)
		BOOST_THROW_EXCEPTION(InvalidGasLimit(gasLimit, _parent.gasLimit * (c_gasLimitBoundDivisor - 1) / c_gasLimitBoundDivisor, _parent.gasLimit * (c_gasLimitBoundDivisor + 1) / c_gasLimitBoundDivisor));

	if (seedHash != calculateSeedHash(_parent))
		BOOST_THROW_EXCEPTION(InvalidSeedHash());

	// Check timestamp is after previous timestamp.
	if (parentHash)
	{
		if (parentHash != _parent.hash)
			BOOST_THROW_EXCEPTION(InvalidParentHash());

		if (timestamp <= _parent.timestamp)
			BOOST_THROW_EXCEPTION(InvalidTimestamp());

		if (number != _parent.number + 1)
			BOOST_THROW_EXCEPTION(InvalidNumber());
	}
}
