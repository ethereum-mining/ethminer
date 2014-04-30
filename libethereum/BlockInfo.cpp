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

#include <libethcore/Common.h>
#include <libethcore/RLP.h>
#include <libethcore/TrieDB.h>
#include "Dagger.h"
#include "Exceptions.h"
#include "State.h"
#include "BlockInfo.h"
using namespace std;
using namespace eth;

BlockInfo* BlockInfo::s_genesis = nullptr;

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

bytes BlockInfo::createGenesisBlock()
{
	RLPStream block(3);
	auto sha3EmptyList = sha3(RLPEmptyList);

	h256 stateRoot;
	{
		BasicMap db;
		TrieDB<Address, BasicMap> state(&db);
		state.init();
		eth::commit(genesisState(), db, state);
		stateRoot = state.root();
	}

	block.appendList(13) << h256() << sha3EmptyList << h160() << stateRoot << h256() << c_genesisDifficulty << 0 << 0 << 1000000 << 0 << (uint)0 << string() << sha3(bytes(1, 42));
	block.appendRaw(RLPEmptyList);
	block.appendRaw(RLPEmptyList);
	return block.out();
}

h256 BlockInfo::headerHashWithoutNonce() const
{
	RLPStream s;
	fillStream(s, false);
	return sha3(s.out());
}

void BlockInfo::fillStream(RLPStream& _s, bool _nonce) const
{
	_s.appendList(_nonce ? 13 : 12) << parentHash << sha3Uncles << coinbaseAddress << stateRoot << transactionsRoot << difficulty << number << minGasPrice << gasLimit << gasUsed << timestamp << extraData;
	if (_nonce)
		_s << nonce;
}

void BlockInfo::populateGenesis()
{
	bytes genesisBlock = createGenesisBlock();
	populate(&genesisBlock);
}

void BlockInfo::populateFromHeader(RLP const& _header)
{
	int field = 0;
	try
	{
		parentHash = _header[field = 0].toHash<h256>();
		sha3Uncles = _header[field = 1].toHash<h256>();
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
	catch (RLPException const&)
	{
		throw InvalidBlockHeaderFormat(field, _header[field].data());
	}

	// check it hashes according to proof of work or that it's the genesis block.
	if (parentHash && !Dagger::verify(headerHashWithoutNonce(), nonce, difficulty))
		throw InvalidBlockNonce(headerHashWithoutNonce(), nonce, difficulty);

	if (gasUsed > gasLimit)
		throw TooMuchGasUsed();

	if (number && extraData.size() > 1024)
		throw ExtraDataTooBig();
}

void BlockInfo::populate(bytesConstRef _block)
{
	hash = eth::sha3(_block);

	RLP root(_block);
	RLP header = root[0];
	if (!header.isList())
		throw InvalidBlockFormat(0, header.data());
	populateFromHeader(header);

	if (!root[1].isList())
		throw InvalidBlockFormat(1, root[1].data());
	if (!root[2].isList())
		throw InvalidBlockFormat(2, root[2].data());
}

void BlockInfo::verifyInternals(bytesConstRef _block) const
{
	RLP root(_block);

	u256 mgp = (u256)-1;

	Overlay db;
	GenericTrieDB<Overlay> t(&db);
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
		throw InvalidTransactionsHash(t.root(), transactionsRoot);

	if (minGasPrice > mgp)
		throw InvalidMinGasPrice(minGasPrice, mgp);

	if (sha3Uncles != sha3(root[2].data()))
		throw InvalidUnclesHash();
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
		return (_parent.gasLimit * (1024 - 1) + (_parent.gasUsed * 6 / 5)) / 1024;
}

u256 BlockInfo::calculateDifficulty(BlockInfo const& _parent) const
{
	if (!parentHash)
		return c_genesisDifficulty;
	else
		return timestamp >= _parent.timestamp + 42 ? _parent.difficulty - (_parent.difficulty >> 10) : (_parent.difficulty + (_parent.difficulty >> 10));
}

void BlockInfo::verifyParent(BlockInfo const& _parent) const
{
	// Check difficulty is correct given the two timestamps.
	if (difficulty != calculateDifficulty(_parent))
		throw InvalidDifficulty();

	if (gasLimit != calculateGasLimit(_parent))
		throw InvalidGasLimit();

	// Check timestamp is after previous timestamp.
	if (parentHash)
	{
		if (parentHash != _parent.hash)
			throw InvalidParentHash();

		if (timestamp < _parent.timestamp)
			throw InvalidTimestamp();

		if (number != _parent.number + 1)
			throw InvalidNumber();
	}
}
