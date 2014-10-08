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
#include <libdevcore/CommonIO.h>

using namespace dev;
using namespace dev::eth;

const char* InvalidBlockFormat::what() const noexcept { return ("Invalid block format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")").c_str(); }
const char* UncleInChain::what() const noexcept { return ("Uncle in block already mentioned: Uncles " + toString(m_uncles) + " (" + m_block.abridged() + ")").c_str(); }
const char* InvalidTransactionsHash::what() const noexcept { return ("Invalid transactions hash:  header says: " + toHex(m_head.ref()) + " block is:" + toHex(m_real.ref())).c_str(); }
const char* InvalidGasLimit::what() const noexcept { return ("Invalid gas limit (provided: " + toString(provided) + " valid:" + toString(valid) + ")").c_str(); }
const char* InvalidMinGasPrice::what() const noexcept { return ("Invalid minimum gas price (provided: " + toString(provided) + " limit:" + toString(limit) + ")").c_str(); }
const char* InvalidNonce::what() const noexcept { return ("Invalid nonce (r: " + toString(required) + " c:" + toString(candidate) + ")").c_str(); }
const char* InvalidBlockNonce::what() const noexcept { return ("Invalid nonce (h: " + toString(h) + " n:" + toString(n) + " d:" + toString(d) + ")").c_str(); }

