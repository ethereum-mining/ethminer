//
// Created by Marek Kotewicz on 27/04/15.
//

#pragma once

#include <v8.h>
#include "JSScope.h"

namespace dev
{
namespace eth
{

class JSV8Env
{
public:
	JSV8Env();
	~JSV8Env();

private:
	v8::Platform* m_platform;
};

v8::Handle<v8::Context> CreateShellContext(v8::Isolate* isolate)
{
	v8::Handle<v8::ObjectTemplate> global = v8::ObjectTemplate::New(isolate);
	return v8::Context::New(isolate, NULL, global);
}

class JSV8DumbScope
{
public:
	JSV8DumbScope(v8::Isolate* _isolate):
			m_isolateScope(_isolate),
			m_handleScope(_isolate),
			m_context(CreateShellContext(_isolate)),
			m_contextScope(m_context)
	{}

	v8::Handle <v8::Context> const& context() const { return m_context; }

private:
	v8::Isolate::Scope m_isolateScope;
	v8::HandleScope m_handleScope;
	v8::Handle <v8::Context> m_context;
	v8::Context::Scope m_contextScope;
};

class JSV8ScopeBase : public JSScope
{
public:
	JSV8ScopeBase();

	virtual ~JSV8ScopeBase();

	const char* evaluate(const char* _cstr) const;

private:
	static JSV8Env s_env;
	v8::Isolate* m_isolate;
	JSV8DumbScope* m_scope;

	virtual const char* formatValue(v8::Handle <v8::Value> const &_value) const;
};

}
}

