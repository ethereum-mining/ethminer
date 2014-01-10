/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file BlockInfo.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include "Dagger.h"
#include "Exceptions.h"
#include "RLP.h"
#include "BlockInfo.h"
using namespace std;
using namespace eth;

BlockInfo* BlockInfo::s_genesis = nullptr;

BlockInfo::BlockInfo()
{
	number = Invalid256;
}

BlockInfo::BlockInfo(bytesConstRef _block, u256 _number)
{
	populate(_block, _number);
}

bytes BlockInfo::createGenesisBlock()
{
	RLPStream block(3);
	auto sha3EmptyList = sha3(RLPEmptyList);
	block.appendList(8) << (uint)0 << sha3EmptyList << (uint)0 << sha3(RLPNull) << sha3EmptyList << ((uint)1 << 36) << (uint)0 << (uint)0;
	block.appendRaw(RLPEmptyList);
	block.appendRaw(RLPEmptyList);
	return block.out();
}

u256 BlockInfo::headerHashWithoutNonce() const
{
	return sha3((RLPStream(7) << toBigEndianString(parentHash) << toBigEndianString(sha3Uncles) << coinbaseAddress << toBigEndianString(stateRoot) << toBigEndianString(sha3Transactions) << difficulty << timestamp).out());
}

void BlockInfo::populateGenesis()
{
	bytes genesisBlock = createGenesisBlock();
	populate(&genesisBlock, 0);
}

void BlockInfo::populate(bytesConstRef _block, u256 _number)
{
	number = _number;

	RLP root(_block);
	try
	{
		RLP header = root[0];
		hash = eth::sha3(_block);
		parentHash = header[0].toInt<u256>();
		sha3Uncles = header[1].toInt<u256>();
		coinbaseAddress = header[2].toInt<u160>();
		stateRoot = header[3].toInt<u256>();
		sha3Transactions = header[4].toInt<u256>();
		difficulty = header[5].toInt<u256>();
		timestamp = header[6].toInt<u256>();
		nonce = header[7].toInt<u256>();
	}
	catch (RLP::BadCast)
	{
		throw InvalidBlockFormat();
	}
}

void BlockInfo::verifyInternals(bytesConstRef _block) const
{
	RLP root(_block);

	if (sha3Transactions != sha3(root[1].data()))
		throw InvalidTransactionsHash();

	if (sha3Uncles != sha3(root[2].data()))
		throw InvalidUnclesHash();

	// check it hashes according to proof of work.
	Dagger d(headerHashWithoutNonce());
	if (d.eval(nonce) >= difficulty)
		throw InvalidNonce();
}

u256 BlockInfo::calculateDifficulty(BlockInfo const& _parent) const
{
	/*
	D(genesis_block) = 2^36
	D(block) =
		if block.timestamp >= block.parent.timestamp + 42: D(block.parent) - floor(D(block.parent) / 1024)
		else:                                              D(block.parent) + floor(D(block.parent) / 1024)
			*/
	if (number == 0)
		return (u256)1 << 36;
	else
		return timestamp >= _parent.timestamp + 42 ? _parent.difficulty - (_parent.difficulty >> 10) : (_parent.difficulty + (_parent.difficulty >> 10));
}

void BlockInfo::verifyParent(BlockInfo const& _parent) const
{
	if (number == 0)
		// Genesis block - no parent.
		return;

	// Check timestamp is after previous timestamp.
	if (_parent.timestamp <= _parent.timestamp)
		throw InvalidTimestamp();

	// Check difficulty is correct given the two timestamps.
	if (difficulty != calculateDifficulty(_parent))
		throw InvalidDifficulty();
}
