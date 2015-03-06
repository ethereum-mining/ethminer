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
/** @file Exceptions.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Exceptions.h"
#include <boost/thread.hpp>
#include <libdevcore/CommonIO.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

InvalidBlockFormat::InvalidBlockFormat(int _f, bytesConstRef _d):
	Exception("Invalid block format: Bad field " + toString(_f) + " (" + toHex(_d) + ")"), f(_f), d(_d.toBytes()) {}

UncleInChain::UncleInChain(h256Set _uncles, h256 _block):
	Exception("Uncle in block already mentioned: Uncles " + toString(_uncles) + " (" + _block.abridged() + ")"), uncles(_uncles), block(_block) {}

InvalidTransactionsHash::InvalidTransactionsHash(h256 _head, h256 _real):
	Exception("Invalid transactions hash:  header says: " + toHex(_head.ref()) + " block is:" + toHex(_real.ref())), head(_head), real(_real) {}

InvalidGasLimit::InvalidGasLimit(u256 _provided, u256 _valid):
	Exception("Invalid gas limit (provided: " + toString(_provided) + " valid:" + toString(_valid) + ")"), provided(_provided), valid(_valid) {}

InvalidMinGasPrice::InvalidMinGasPrice(u256 _provided, u256 _limit):
	Exception("Invalid minimum gas price (provided: " + toString(_provided) + " limit:" + toString(_limit) + ")"), provided(_provided), limit(_limit) {}

InvalidNonce::InvalidNonce(u256 _required, u256 _candidate):
	Exception("Invalid nonce (r: " + toString(_required) + " c:" + toString(_candidate) + ")"), required(_required), candidate(_candidate) {}

InvalidBlockNonce::InvalidBlockNonce(h256 _h, h256 _n, u256 _d):
	Exception("Invalid nonce (h: " + toString(h) + " n:" + toString(n) + " d:" + toString(d) + ")"), h(_h), n(_n), d(_d) {}

