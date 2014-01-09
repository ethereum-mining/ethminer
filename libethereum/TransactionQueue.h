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
/** @file TransactionQueue.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "Common.h"

namespace eth
{

class BlockChain;

/**
 * @brief A queue of Transactions, each stored as RLP.
 */
class TransactionQueue
{
public:
	bool attemptImport(bytes const& _block) { try { import(_block); return true; } catch (...) { return false; } }

	void import(bytes const& _block);

	void drop(u256 _txHash) { m_data.erase(_txHash); }

	std::map<u256, bytes> const& transactions() const { return m_data; }

private:
	std::map<u256, bytes> m_data;	///< the queue.
};

}


