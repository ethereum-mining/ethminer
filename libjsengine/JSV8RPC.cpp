//
// Created by Marek Kotewicz on 04/05/15.
//

#include "libjsconsole/JSConsoleResources.hpp"
#include "JSV8RPC.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace eth
{

void JSV8RPCSend(v8::FunctionCallbackInfo<v8::Value> const& args)
{
	v8::Local<v8::String> JSON = v8::String::NewFromUtf8(args.GetIsolate(), "JSON");
	v8::Local<v8::String> parse = v8::String::NewFromUtf8(args.GetIsolate(), "parse");
	v8::Local<v8::String> stringify = v8::String::NewFromUtf8(args.GetIsolate(), "stringify");
	v8::Handle<v8::Object> jsonObject = v8::Handle<v8::Object>::Cast(args.GetIsolate()->GetCurrentContext()->Global()->Get(JSON));
	v8::Handle<v8::Function> parseFunc = v8::Handle<v8::Function>::Cast(jsonObject->Get(parse));
	v8::Handle<v8::Function> stringifyFunc = v8::Handle<v8::Function>::Cast(jsonObject->Get(stringify));

	v8::Local<v8::Object> self = args.Holder();
	v8::Local<v8::External> wrap = v8::Local<v8::External>::Cast(self->GetInternalField(0));
	JSV8RPC* that = static_cast<JSV8RPC*>(wrap->Value());
	v8::Local<v8::Value> vals[1] = {args[0]->ToObject()};
	v8::Local<v8::Value> stringifiedArg = stringifyFunc->Call(stringifyFunc, 1, vals);
	v8::String::Utf8Value str(stringifiedArg);
	that->onSend(*str);

	v8::Local<v8::Value> values[1] = {v8::String::NewFromUtf8(args.GetIsolate(), that->m_lastResponse)};
	args.GetReturnValue().Set(parseFunc->Call(parseFunc, 1, values));
}

}
}

JSV8RPC::JSV8RPC(JSV8Engine const &_engine): m_engine(_engine)
{
	v8::HandleScope scope(m_engine.context()->GetIsolate());
	v8::Local<v8::ObjectTemplate> rpcTemplate = v8::ObjectTemplate::New(m_engine.context()->GetIsolate());
	rpcTemplate->SetInternalFieldCount(1);
	rpcTemplate->Set(v8::String::NewFromUtf8(m_engine.context()->GetIsolate(), "send"),
	                 v8::FunctionTemplate::New(m_engine.context()->GetIsolate(), JSV8RPCSend));
	rpcTemplate->Set(v8::String::NewFromUtf8(m_engine.context()->GetIsolate(), "sendAsync"),
	                 v8::FunctionTemplate::New(m_engine.context()->GetIsolate(), JSV8RPCSend));

	v8::Local<v8::Object> obj = rpcTemplate->NewInstance();
	obj->SetInternalField(0, v8::External::New(m_engine.context()->GetIsolate(), this));

	v8::Local<v8::String> web3 = v8::String::NewFromUtf8(m_engine.context()->GetIsolate(), "web3");
	v8::Local<v8::String> setProvider = v8::String::NewFromUtf8(m_engine.context()->GetIsolate(), "setProvider");
	v8::Handle<v8::Object> web3object = v8::Handle<v8::Object>::Cast(m_engine.context()->Global()->Get(web3));
	v8::Handle<v8::Function> func = v8::Handle<v8::Function>::Cast(web3object->Get(setProvider));
	v8::Local<v8::Value> values[1] = {obj};
	func->Call(func, 1, values);

	m_lastResponse = R"(
	{
		"id": 1,
		"jsonrpc": "2.0",
		"error": "Uninitalized JSV8RPC!"
	}
	)";
}
