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
/** @file JSV8Engine.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#include <memory>
#include "JSV8Engine.h"
#include "libjsengine/JSEngineResources.hpp"

using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace eth
{

static char const* toCString(v8::String::Utf8Value const& _value)
{
	if (*_value)
		return *_value;
	throw JSPrintException();
}

// from:        https://github.com/v8/v8-git-mirror/blob/master/samples/shell.cc
// v3.15 from:  https://chromium.googlesource.com/v8/v8.git/+/3.14.5.9/samples/shell.cc
void reportException(v8::TryCatch* _tryCatch)
{
	v8::HandleScope handle_scope;
	v8::String::Utf8Value exception(_tryCatch->Exception());
	char const* exceptionString = toCString(exception);
	v8::Handle<v8::Message> message = _tryCatch->Message();

	// V8 didn't provide any extra information about this error; just
	// print the exception.
	if (message.IsEmpty())
		printf("%s\n", exceptionString);
	else
	{
		// Print (filename):(line number): (message).
		v8::String::Utf8Value filename(message->GetScriptResourceName());
		char const* filenameString = toCString(filename);
		int linenum = message->GetLineNumber();
		printf("%s:%i: %s\n", filenameString, linenum, exceptionString);

		// Print line of source code.
		v8::String::Utf8Value sourceline(message->GetSourceLine());
		char const* sourcelineString = toCString(sourceline);
		printf("%s\n", sourcelineString);

		// Print wavy underline (GetUnderline is deprecated).
		int start = message->GetStartColumn();
		for (int i = 0; i < start; i++)
			printf(" ");

		int end = message->GetEndColumn();

		for (int i = start; i < end; i++)
			printf("^");

		printf("\n");

		v8::String::Utf8Value stackTrace(_tryCatch->StackTrace());
		if (stackTrace.length() > 0)
		{
			char const* stackTraceString = toCString(stackTrace);
			printf("%s\n", stackTraceString);
		}
	}
}

class JSV8Env
{
public:
	static JSV8Env& getInstance()
	{
		static JSV8Env s_env;
		return s_env;
	}

	~JSV8Env()
	{
		v8::V8::Dispose();
	}
};

class JSV8Scope
{
public:
	JSV8Scope():
		m_handleScope(),
		m_context(v8::Context::New(NULL, v8::ObjectTemplate::New())),
		m_contextScope(m_context)
	{
		m_context->Enter();
	}

	~JSV8Scope()
	{
		m_context->Exit();
		m_context.Dispose();
	}

	v8::Persistent <v8::Context> const& context() const { return m_context; }

private:
	v8::HandleScope m_handleScope;
	v8::Persistent <v8::Context> m_context;
	v8::Context::Scope m_contextScope;
};

}
}

JSString JSV8Value::toString() const
{
	if (m_value.IsEmpty())
		return "";

	else if (m_value->IsUndefined())
		return "undefined";

	v8::String::Utf8Value str(m_value);
	return toCString(str);
}

JSV8Engine::JSV8Engine(): m_env(JSV8Env::getInstance()), m_scope(new JSV8Scope())
{
	JSEngineResources resources;
	string common = resources.loadResourceAsString("common");
	string web3 = resources.loadResourceAsString("web3");
	string admin = resources.loadResourceAsString("admin");
	eval(common.c_str());
	eval(web3.c_str());
	eval("web3 = require('web3');");
	eval(admin.c_str());
}

JSV8Engine::~JSV8Engine()
{
	delete m_scope;
}

JSV8Value JSV8Engine::eval(char const* _cstr) const
{
	v8::TryCatch tryCatch;
	v8::Local<v8::String> source = v8::String::New(_cstr);
	v8::Local<v8::String> name(v8::String::New("(shell)"));
	v8::ScriptOrigin origin(name);
	v8::Handle<v8::Script> script = v8::Script::Compile(source, &origin);

	// Make sure to wrap the exception in a new handle because
	// the handle returned from the TryCatch is destroyed
	if (script.IsEmpty())
	{
		reportException(&tryCatch);
		return v8::Exception::Error(v8::Local<v8::String>::New(tryCatch.Message()->Get()));
	}

	auto result = script->Run();

	if (result.IsEmpty())
	{
		reportException(&tryCatch);
		return v8::Exception::Error(v8::Local<v8::String>::New(tryCatch.Message()->Get()));
	}

	return result;
}

v8::Handle<v8::Context> const& JSV8Engine::context() const
{
	return m_scope->context();
}
