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

BlockInfo::BlockInfo(bytesConstRef _block, Strictness _s)
{
	populate(_block, _s);
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
	mixHash = h256();
	nonce = Nonce();
	m_seedHash = h256();
	hash = headerHash(WithNonce);
}

h256 BlockInfo::seedHash() const
{
	if (!m_seedHash)
		for (u256 n = number; n >= c_epochDuration; n -= c_epochDuration)
			m_seedHash = sha3(m_seedHash);
	return m_seedHash;
}

BlockInfo BlockInfo::fromHeader(bytesConstRef _block, Strictness _s)
{
	BlockInfo ret;
	ret.populateFromHeader(RLP(_block), _s);
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
	_s.appendList(_n == WithNonce ? 15 : 13)
		<< parentHash << sha3Uncles << coinbaseAddress << stateRoot << transactionsRoot << receiptsRoot << logBloom
		<< difficulty << number << gasLimit << gasUsed << timestamp << extraData;
	if (_n == WithNonce)
		_s << mixHash << nonce;
}

h256 BlockInfo::headerHash(bytesConstRef _block)
{
	return sha3(RLP(_block)[0].data());
}

void BlockInfo::populateFromHeader(RLP const& _header, Strictness _s)
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
		mixHash = _header[field = 13].toHash<h256>(RLP::VeryStrict);
		nonce = _header[field = 14].toHash<Nonce>(RLP::VeryStrict);
	}

	catch (Exception const& _e)
	{
		_e << errinfo_name("invalid block header format") << BadFieldError(field, toHex(_header[field].data().toBytes()));
		throw;
	}

	// check it hashes according to proof of work or that it's the genesis block.
	if (_s == CheckEverything && parentHash && !ProofOfWork::verify(*this))
		BOOST_THROW_EXCEPTION(InvalidBlockNonce() << errinfo_hash256(headerHash(WithoutNonce)) << errinfo_nonce(nonce) << errinfo_difficulty(difficulty));

	if (_s != CheckNothing)
	{
		if (gasUsed > gasLimit)
			BOOST_THROW_EXCEPTION(TooMuchGasUsed() << RequirementError(bigint(gasLimit), bigint(gasUsed)) );

		if (difficulty < c_minimumDifficulty)
			BOOST_THROW_EXCEPTION(InvalidDifficulty() << RequirementError(bigint(c_minimumDifficulty), bigint(difficulty)) );

		if (gasLimit < c_minGasLimit)
			BOOST_THROW_EXCEPTION(InvalidGasLimit() << RequirementError(bigint(c_minGasLimit), bigint(gasLimit)) );

		if (number && extraData.size() > c_maximumExtraDataSize)
			BOOST_THROW_EXCEPTION(ExtraDataTooBig() << RequirementError(bigint(c_maximumExtraDataSize), bigint(extraData.size())));
	}
}

void BlockInfo::populate(bytesConstRef _block, Strictness _s)
{
	RLP root(_block);
	RLP header = root[0];

	if (!header.isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block header needs to be a list") << BadFieldError(0, header.data().toString()));
	populateFromHeader(header, _s);

	if (!root[1].isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block transactions need to be a list") << BadFieldError(1, root[1].data().toString()));
	if (!root[2].isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block uncles need to be a list") << BadFieldError(2, root[2].data().toString()));
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
		BOOST_THROW_EXCEPTION(InvalidTransactionsHash() << HashMismatchError(t.root(), transactionsRoot));

	if (sha3Uncles != sha3(root[2].data()))
		BOOST_THROW_EXCEPTION(InvalidUnclesHash());
}

void BlockInfo::populateFromParent(BlockInfo const& _parent)
{
	stateRoot = _parent.stateRoot;
	parentHash = _parent.hash;
	number = _parent.number + 1;
	gasLimit = selectGasLimit(_parent);
	gasUsed = 0;
	difficulty = calculateDifficulty(_parent);
}

u256 BlockInfo::selectGasLimit(BlockInfo const& _parent) const
{
	if (!parentHash)
		return c_genesisGasLimit;
	else
		// target minimum of 3141592
		return max<u256>(max<u256>(c_minGasLimit, 3141592), (_parent.gasLimit * (c_gasLimitBoundDivisor - 1) + (_parent.gasUsed * 6 / 5)) / c_gasLimitBoundDivisor);
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

	if (gasLimit < c_minGasLimit ||
		gasLimit < _parent.gasLimit * (c_gasLimitBoundDivisor - 1) / c_gasLimitBoundDivisor ||
		gasLimit > _parent.gasLimit * (c_gasLimitBoundDivisor + 1) / c_gasLimitBoundDivisor)
		BOOST_THROW_EXCEPTION(InvalidGasLimit() << errinfo_min((bigint)_parent.gasLimit * (c_gasLimitBoundDivisor - 1) / c_gasLimitBoundDivisor) << errinfo_got((bigint)gasLimit) << errinfo_max((bigint)_parent.gasLimit * (c_gasLimitBoundDivisor + 1) / c_gasLimitBoundDivisor));

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
