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
/** @file Common.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include "Exceptions.h"
#include "Log.h"
using namespace std;
using namespace dev;

namespace dev
{

char const* Version = "0.9.23";

const u256 UndefinedU256 = ~(u256)0;

void HasInvariants::checkInvariants() const
{
	if (!invariants())
		BOOST_THROW_EXCEPTION(FailedInvariant());
}

struct TimerChannel: public LogChannel { static const char* name(); static const int verbosity = 0; };

#ifdef _WIN32
const char* TimerChannel::name() { return EthRed " ! "; }
#else
const char* TimerChannel::name() { return EthRed " ⚡ "; }
#endif

TimerHelper::~TimerHelper()
{
	auto e = m_t.elapsed();
	if (!m_ms || e * 1000 > m_ms)
		clog(TimerChannel) << m_id << e << "s";
}

}

