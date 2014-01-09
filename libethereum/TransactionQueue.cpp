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
/** @file TransactionQueue.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Transaction.h"
#include "TransactionQueue.h"
using namespace std;
using namespace eth;

void TransactionQueue::import(bytes const& _block)
{
	// Check if we already know this transaction.
	u256 h = sha3(_block);
	if (m_data.count(h))
		return;

	// Check validity of _block as a transaction. To do this we just deserialise and attempt to determine the sender. If it doesn't work, the signature is bad.
	// The transaction's nonce may yet be invalid (or, it could be "valid" but we may be missing a marginally older transaction).
	Transaction t(_block);
	t.sender();

	// If valid, append to blocks.
	m_data[h] = _block;
}
