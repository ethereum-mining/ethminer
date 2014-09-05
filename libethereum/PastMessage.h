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
/** @file PastMessage.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libethential/Common.h>
#include <libethcore/CommonEth.h>
#include "Manifest.h"

namespace dev
{
namespace eth
{

struct PastMessage
{
	PastMessage(Manifest const& _m, std::vector<unsigned> _path, Address _o): to(_m.to), from(_m.from), value(_m.value), input(_m.input), output(_m.output), path(_path), origin(_o) {}

	PastMessage& polish(h256 _b, u256 _ts, unsigned _n, Address _coinbase) { block = _b; timestamp = _ts; number = _n; coinbase = _coinbase; return *this; }

	Address to;					///< The receiving address of the transaction. Address() in the case of a creation.
	Address from;				///< The receiving address of the transaction. Address() in the case of a creation.
	u256 value;					///< The value associated with the call.
	bytes input;				///< The data associated with the message, or the initialiser if it's a creation transaction.
	bytes output;				///< The data returned by the message, or the body code if it's a creation transaction.

	std::vector<unsigned> path;	///< Call path into the block transaction. size() is always > 0. First item is the transaction index in the block.
	Address origin;				///< Originating sender of the transaction.
	Address coinbase;			///< Block coinbase.
	h256 block;					///< Block hash.
	u256 timestamp;				///< Block timestamp.
	unsigned number;			///< Block number.
};

typedef std::vector<PastMessage> PastMessages;

}
}
