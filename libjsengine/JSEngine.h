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
/** @file JSEngine.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#pragma once
#include <exception>

/// Do not use libstd headers here, it will break on MacOS.

namespace dev
{
namespace eth
{

class JSException: public std::exception {};
#if defined(_MSC_VER)
class JSPrintException: public JSException { char const* what() const { return "Cannot print expression!"; } };
#else
class JSPrintException: public JSException { char const* what() const noexcept { return "Cannot print expression!"; } };
#endif

class JSString
{
public:
	JSString(char const* _cstr);
	~JSString();
	char const* cstr() const { return m_cstr; }

private:
	char* m_cstr;
};

class JSValue
{
public:
	virtual JSString toString() const = 0;
};

template <typename T>
class JSEngine
{
public:
	// should be used to evalute javascript expression
	virtual T eval(char const* _cstr) const = 0;
};

}
}
