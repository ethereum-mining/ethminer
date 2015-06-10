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
/** @file VerfiedBlock.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */


#include <libdevcore/Common.h>
#include <libethcore/BlockInfo.h>

#pragma once

namespace dev
{
namespace eth
{

class Transaction;

/// @brief Verified block info, does not hold block data, but a reference instead
struct VerifiedBlockRef
{
	bytesConstRef block; 					///<  Block data reference
	BlockInfo info;							///< Prepopulated block info
	std::vector<Transaction> transactions;	///< Verified list of block transactions
};

/// @brief Verified block info, combines block data and verified info/transactions
struct VerifiedBlock
{
	VerifiedBlockRef verified;				///< Verified block structures
	bytes blockData;						///< Block data
};

using VerifiedBlocks = std::vector<VerifiedBlock>;

}
}
