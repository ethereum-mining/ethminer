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

class JSV8Engine : public JSEngine
{
public:
	JSV8Engine();
	virtual ~JSV8Engine();
	v8::Handle<v8::Value> eval(const char* _cstr) const;
	const char* evaluate(const char* _cstr) const;

private:
	static JSV8Env s_env;
	v8::Isolate* m_isolate;
	JSV8Scope* m_scope;

protected:
	v8::Handle<v8::Context> const& context() const;
};

}
}

