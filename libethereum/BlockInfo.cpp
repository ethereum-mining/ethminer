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

void BlockInfo::populateAndVerify(bytesConstRef _block, u256 _number)
{
	number = _number;

	RLP root(_block);
	try
	{
		RLP header = root[0];
		hash = eth::sha256(_block);
		parentHash = header[0].toFatInt();
		sha256Uncles = header[1].toFatInt();
		coinbaseAddress = header[2].toFatInt();
		sha256Transactions = header[3].toFatInt();
		difficulty = header[4].toFatInt();
		timestamp = header[5].toFatInt();
		nonce = header[6].toFatInt();
	}
	catch (RLP::BadCast)
	{
		throw InvalidBlockFormat();
	}

	if (sha256Transactions != sha256(root[1].data()))
		throw InvalidTransactionsHash();

	if (sha256Uncles != sha256(root[2].data()))
		throw InvalidUnclesHash();

	// TODO: check timestamp.
	// TODO: check difficulty against timestamp.
	// TODO: check proof of work.

	// TODO: check each transaction.
}
