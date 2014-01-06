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
#include "sha256.h"
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

bytes BlockInfo::createGenesisBlock()
{
	RLPStream block(3);
	auto sha256EmptyList = sha256(RLPEmptyList);
	block.appendList(7) << (uint)0 << sha256EmptyList << (uint)0 << sha256EmptyList << (uint)0 << (uint)0 << (uint)0;
	block.appendRaw(RLPEmptyList);
	block.appendRaw(RLPEmptyList);
	return block.out();
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
		hash = eth::sha256(_block);
		parentHash = header[0].toInt<u256>();
		sha256Uncles = header[1].toInt<u256>();
		coinbaseAddress = header[2].toInt<u160>();
		sha256Transactions = header[3].toInt<u256>();
		difficulty = header[4].toInt<uint>();
		timestamp = header[5].toInt<u256>();
		nonce = header[6].toInt<u256>();
	}
	catch (RLP::BadCast)
	{
		throw InvalidBlockFormat();
	}
}

void BlockInfo::verify(bytesConstRef _block, u256 _number)
{
	populate(_block, _number);

	RLP root(_block);
	if (sha256Transactions != sha256(root[1].data()))
		throw InvalidTransactionsHash();

	if (sha256Uncles != sha256(root[2].data()))
		throw InvalidUnclesHash();

	// TODO: check timestamp.
	// TODO: check difficulty against timestamp.
	// TODO: check proof of work.

	// TODO: check each transaction - allow coinbaseAddress for the miner fees, but everything else must be exactly how we would do it.
}
