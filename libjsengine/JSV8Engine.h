//
// Created by Marek Kotewicz on 27/04/15.
//

#pragma once

#include <v8.h>
#include "JSEngine.h"

namespace dev
{
namespace eth
{

class JSV8Env;
class JSV8Scope;

class JSV8Value : public JSValue
{
public:
	JSV8Value(v8::Handle<v8::Value> _value): m_value(_value) {}
	const char* asCString() const;

	v8::Handle<v8::Value> const& value() const { return m_value; }
private:
	v8::Handle<v8::Value> m_value;
};

class JSV8Engine : public JSEngine<JSV8Value>
{
public:
	JSV8Engine();
	virtual ~JSV8Engine();
	JSV8Value eval(const char* _cstr) const;
	v8::Handle<v8::Context> const& context() const;

private:
	static JSV8Env s_env;
	v8::Isolate* m_isolate;
	JSV8Scope* m_scope;
};

}
}
