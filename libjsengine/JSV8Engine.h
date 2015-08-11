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
/** @file JSV8Engine.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <v8.h>
#pragma clang diagnostic pop
#include "JSEngine.h"

/// Do not use libstd headers here, it will break on MacOS.

namespace dev
{
namespace eth
{

class JSV8Env;
class JSV8Scope;

class JSV8Value: public JSValue
{
public:
	JSV8Value(v8::Handle<v8::Value> _value): m_value(_value) {}
	JSString toString() const;
	v8::Handle<v8::Value> const& value() const { return m_value; }

private:
	v8::Handle<v8::Value> m_value;
};

class JSV8Engine: public JSEngine<JSV8Value>
{
public:
	JSV8Engine();
	virtual ~JSV8Engine();
	JSV8Value eval(char const* _cstr) const;
	v8::Handle<v8::Context> const& context() const;

private:
	JSV8Scope* m_scope;
};

}
}
