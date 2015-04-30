//
// Created by Marek Kotewicz on 28/04/15.
//

#include <string>
#include "JSV8Printer.h"
#include "libjsengine/JSEngineResources.hpp"
#include "libjsengine/t.h"

using namespace std;
using namespace dev;
using namespace eth;

JSV8Printer::JSV8Printer(JSV8Engine const& _engine): m_engine(_engine)
{
	JSEngineResources resources;
	string prettyPrint = resources.loadResourceAsString("pretty_print");
	m_engine.eval(prettyPrint.c_str());
}

const char* JSV8Printer::prettyPrint(JSV8Value const& _value) const
{
	v8::HandleScope handleScope(m_engine.context()->GetIsolate());
	v8::Local<v8::String> pp = v8::String::NewFromUtf8(m_engine.context()->GetIsolate(), "prettyPrint");
	v8::Handle<v8::Function> func = v8::Handle<v8::Function>::Cast(m_engine.context()->Global()->Get(pp));
	v8::Local<v8::Value> values[1] = {_value.value()};
	v8::Local<v8::Value> res = v8::Local<v8::Value>::Cast(func->Call(func, 1, values));
	v8::String::Utf8Value str(res);
	return *str ? *str : "<pretty print conversion failed>";
}
