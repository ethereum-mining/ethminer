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
/** @file JSV8Printer.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#pragma once

#include "JSPrinter.h"
#include "JSV8Engine.h"

/// Do not use libstd headers here, it will break on MacOS.

namespace dev
{
namespace eth
{

class JSV8Printer: public JSPrinter<JSV8Value>
{
public:
	JSV8Printer(JSV8Engine const& _engine);
	JSString prettyPrint(JSV8Value const& _value) const;
private:
	JSV8Engine const& m_engine;
};

}
}
