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

#if !ETH_LANGUAGES

#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libdevcrypto/TrieDB.h>
#include "ProofOfWork.h"
#include "Exceptions.h"
#include "BlockInfo.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

u256 dev::eth::c_genesisDifficulty = (u256)1 << 17;

BlockInfo::BlockInfo(): timestamp(Invalid256)
{
}

BlockInfo::BlockInfo(bytesConstRef _block)
{
	populate(_block);
}

BlockInfo BlockInfo::fromHeader(bytesConstRef _block)
{
	BlockInfo ret;
	ret.populateFromHeader(RLP(_block));
	return ret;
}

h256 BlockInfo::headerHashWithoutNonce() const
{
	RLPStream s;
	fillStream(s, false);
	return sha3(s.out());
}

auto static const c_sha3EmptyList = sha3(RLPEmptyList);

void BlockInfo::fillStream(RLPStream& _s, bool _nonce) const
{
	_s.appendList(_nonce ? 13 : 12) << parentHash;
	_s.append(sha3Uncles == c_sha3EmptyList ? h256() : sha3Uncles, false, true);
	_s << coinbaseAddress;
	_s.append(stateRoot, false, true).append(transactionsRoot, false, true);
	_s << difficulty << number << minGasPrice << gasLimit << gasUsed << timestamp << extraData;
	if (_nonce)
		_s << nonce;
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
		parentHash = _header[field = 0].toHash<h256>();
		sha3Uncles = _header[field = 1].toHash<h256>();
		if (sha3Uncles == h256())
			sha3Uncles = c_sha3EmptyList;
		coinbaseAddress = _header[field = 2].toHash<Address>();
		stateRoot = _header[field = 3].toHash<h256>();
		transactionsRoot = _header[field = 4].toHash<h256>();
		difficulty = _header[field = 5].toInt<u256>();
		number = _header[field = 6].toInt<u256>();
		minGasPrice = _header[field = 7].toInt<u256>();
		gasLimit = _header[field = 8].toInt<u256>();
		gasUsed = _header[field = 9].toInt<u256>();
		timestamp = _header[field = 10].toInt<u256>();
		extraData = _header[field = 11].toBytes();
		nonce = _header[field = 12].toHash<h256>();
	}

	catch (Exception & _e)
	{
		_e << errinfo_name("invalid block header format") << BadFieldError(field, toHex(_header[field].data().toBytes()));
		throw;
	}

	// check it hashes according to proof of work or that it's the genesis block.
	if (_checkNonce && parentHash && !ProofOfWork::verify(headerHashWithoutNonce(), nonce, difficulty))
		BOOST_THROW_EXCEPTION(InvalidBlockNonce(headerHashWithoutNonce(), nonce, difficulty));

	if (gasUsed > gasLimit)
		BOOST_THROW_EXCEPTION(TooMuchGasUsed());

	if (number && extraData.size() > 1024)
		BOOST_THROW_EXCEPTION(ExtraDataTooBig());
}

void BlockInfo::populate(bytesConstRef _block, bool _checkNonce)
{
	RLP root(_block);
	RLP header = root[0];

	if (!header.isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat(0,header.data()));
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
		u256 gp = tr[0][1].toInt<u256>();
		mgp = min(mgp, gp);
		++i;
	}
	if (transactionsRoot != t.root())
		BOOST_THROW_EXCEPTION(InvalidTransactionsHash(t.root(), transactionsRoot));

	if (minGasPrice > mgp)
		BOOST_THROW_EXCEPTION(InvalidMinGasPrice(minGasPrice, mgp));

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
}

u256 BlockInfo::calculateGasLimit(BlockInfo const& _parent) const
{
	if (!parentHash)
		return 1000000;
	else
		return max<u256>(125000, (_parent.gasLimit * (1024 - 1) + (_parent.gasUsed * 6 / 5)) / 1024);
}

u256 BlockInfo::calculateDifficulty(BlockInfo const& _parent) const
{
	if (!parentHash)
		return c_genesisDifficulty;
	else
		return timestamp >= _parent.timestamp + 5 ? _parent.difficulty - (_parent.difficulty >> 10) : (_parent.difficulty + (_parent.difficulty >> 10));
}

void BlockInfo::verifyParent(BlockInfo const& _parent) const
{	// Check difficulty is correct given the two timestamps.
	if (difficulty != calculateDifficulty(_parent))
		BOOST_THROW_EXCEPTION(InvalidDifficulty());

	if (gasLimit != calculateGasLimit(_parent))
		BOOST_THROW_EXCEPTION(InvalidGasLimit(gasLimit, calculateGasLimit(_parent)));

	// Check timestamp is after previous timestamp.
	if (parentHash)
	{
		if (parentHash != _parent.hash)
			BOOST_THROW_EXCEPTION(InvalidParentHash());

		if (timestamp < _parent.timestamp)
			BOOST_THROW_EXCEPTION(InvalidTimestamp());

		if (number != _parent.number + 1)
			BOOST_THROW_EXCEPTION(InvalidNumber());
	}
}

#endif
