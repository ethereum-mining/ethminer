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

#if ALL_COMPILERS_ARE_CPP11
#define ETH_RETURN_STRING(S) thread_local static string s_what; s_what = S; return s_what.c_str();
#elsif USE_BOOST_TLS
static boost::thread_specific_ptr<string> g_exceptionMessage;
#define ETH_RETURN_STRING(S) if (!g_exceptionMessage.get()); g_exceptionMessage.reset(new string); *g_exceptionMessage.get() = S; return g_exceptionMessage.get()->c_str();
#else
#define ETH_RETURN_STRING(S) m_message = S; return m_message.c_str();
#endif

const char* InvalidBlockFormat::what() const noexcept { ETH_RETURN_STRING("Invalid block format: Bad field " + toString(m_f) + " (" + toHex(m_d) + ")"); }
const char* UncleInChain::what() const noexcept { ETH_RETURN_STRING("Uncle in block already mentioned: Uncles " + toString(m_uncles) + " (" + m_block.abridged() + ")"); }
const char* InvalidTransactionsHash::what() const noexcept { ETH_RETURN_STRING("Invalid transactions hash:  header says: " + toHex(m_head.ref()) + " block is:" + toHex(m_real.ref())); }
const char* InvalidGasLimit::what() const noexcept { ETH_RETURN_STRING("Invalid gas limit (provided: " + toString(provided) + " valid:" + toString(valid) + ")"); }
const char* InvalidMinGasPrice::what() const noexcept { ETH_RETURN_STRING("Invalid minimum gas price (provided: " + toString(provided) + " limit:" + toString(limit) + ")"); }
const char* InvalidNonce::what() const noexcept { ETH_RETURN_STRING("Invalid nonce (r: " + toString(required) + " c:" + toString(candidate) + ")"); }
const char* InvalidBlockNonce::what() const noexcept { ETH_RETURN_STRING("Invalid nonce (h: " + toString(h) + " n:" + toString(n) + " d:" + toString(d) + ")"); }

