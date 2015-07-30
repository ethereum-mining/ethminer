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
/** @file JSV8RPC.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#include "JSV8RPC.h"

using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace eth
{

v8::Handle<v8::Value> JSV8RPCSend(v8::Arguments const& _args)
{
	v8::Local<v8::String> JSON = v8::String::New("JSON");
	v8::Local<v8::String> parse = v8::String::New("parse");
	v8::Local<v8::String> stringify = v8::String::New("stringify");
	v8::Handle<v8::Object> jsonObject = v8::Handle<v8::Object>::Cast(v8::Context::GetCurrent()->Global()->Get(JSON));
	v8::Handle<v8::Function> parseFunc = v8::Handle<v8::Function>::Cast(jsonObject->Get(parse));
	v8::Handle<v8::Function> stringifyFunc = v8::Handle<v8::Function>::Cast(jsonObject->Get(stringify));

	v8::Local<v8::Object> self = _args.Holder();
	v8::Local<v8::External> wrap = v8::Local<v8::External>::Cast(self->GetInternalField(0));
	JSV8RPC* that = static_cast<JSV8RPC*>(wrap->Value());
	v8::Local<v8::Value> vals[1] = {_args[0]->ToObject()};
	v8::Local<v8::Value> stringifiedArg = stringifyFunc->Call(stringifyFunc, 1, vals);
	v8::String::Utf8Value str(stringifiedArg);
	that->onSend(*str);

	v8::Local<v8::Value> values[1] = {v8::String::New(that->lastResponse())};
	return parseFunc->Call(parseFunc, 1, values);
}

v8::Handle<v8::Value> JSV8RPCSendAsync(v8::Arguments const& _args)
{
	// This is synchronous, but uses the callback-interface.

	auto parsed = v8::Local<v8::Value>::New(JSV8RPCSend(_args));
	v8::Handle<v8::Function> callback = v8::Handle<v8::Function>::Cast(_args[1]);
	v8::Local<v8::Value> callbackArgs[2] = {v8::Local<v8::Value>::New(v8::Null()), parsed};
	callback->Call(callback, 2, callbackArgs);

	return v8::Undefined();
}

}
}

JSV8RPC::JSV8RPC(JSV8Engine const& _engine): m_engine(_engine)
{
	v8::HandleScope scope;
	v8::Local<v8::ObjectTemplate> rpcTemplate = v8::ObjectTemplate::New();
	rpcTemplate->SetInternalFieldCount(1);
	rpcTemplate->Set(
		v8::String::New("send"),
		v8::FunctionTemplate::New(JSV8RPCSend)
	);
	rpcTemplate->Set(
		v8::String::New("sendAsync"),
		v8::FunctionTemplate::New(JSV8RPCSendAsync)
	);

	v8::Local<v8::Object> obj = rpcTemplate->NewInstance();
	obj->SetInternalField(0, v8::External::New(this));

	v8::Local<v8::String> web3 = v8::String::New("web3");
	v8::Local<v8::String> setProvider = v8::String::New("setProvider");
	v8::Handle<v8::Object> web3object = v8::Handle<v8::Object>::Cast(m_engine.context()->Global()->Get(web3));
	v8::Handle<v8::Function> func = v8::Handle<v8::Function>::Cast(web3object->Get(setProvider));
	v8::Local<v8::Value> values[1] = {obj};
	func->Call(func, 1, values);
}
