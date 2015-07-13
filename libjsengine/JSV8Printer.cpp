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
/** @file JSV8Printer.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#include <string>
#include "JSV8Printer.h"
#include "libjsengine/JSEngineResources.hpp"

using namespace std;
using namespace dev;
using namespace eth;

JSV8Printer::JSV8Printer(JSV8Engine const& _engine): m_engine(_engine)
{
	JSEngineResources resources;
	string prettyPrint = resources.loadResourceAsString("pretty_print");
	m_engine.eval(prettyPrint.c_str());
}

JSString JSV8Printer::prettyPrint(JSV8Value const& _value) const
{
	v8::Local<v8::String> pp = v8::String::New("prettyPrint");
	v8::Handle<v8::Function> func = v8::Handle<v8::Function>::Cast(m_engine.context()->Global()->Get(pp));
	v8::Local<v8::Value> values[1] = {v8::Local<v8::Value>::New(_value.value())};
	v8::Local<v8::Value> res = func->Call(func, 1, values);
	v8::String::Utf8Value str(res);
	if (*str)
		return *str;
	throw JSPrintException();
}
