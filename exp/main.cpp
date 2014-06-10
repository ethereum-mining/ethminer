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
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Ethereum client.
 */

#include <libethsupport/Log.h>
#include <libethsupport/Common.h>
#include <libethsupport/CommonData.h>
#include "BuildInfo.h"
using namespace std;
using namespace eth;

int main(int, char**)
{
    u256 z = 0;
    u256 s = 7;
    u256 ms = z - s;
    s256 ams = -7;
    s256 sms = u2s(ms);
    cnote << sms;
    cnote << ams;
    cnote << ms;
    u256 t = 3;
    s256 st = u2s(t);
    cnote << ms << t << (sms % t) << sms << st << (s2u(sms % st) + 70);
	return 0;
}
