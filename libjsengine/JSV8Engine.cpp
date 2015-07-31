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
#include "JSV8Printer.h"
#include "libjsengine/JSEngineResources.hpp"
#include "BuildInfo.h"

#define TO_STRING_HELPER(s) #s
#define TO_STRING(s) TO_STRING_HELPER(s)

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace v8;

namespace dev
{
namespace eth
{

static char const* toCString(String::Utf8Value const& _value)
{
	if (*_value)
		return *_value;
	throw JSPrintException();
}

// from:        https://github.com/v8/v8-git-mirror/blob/master/samples/shell.cc
// v3.15 from:  https://chromium.googlesource.com/v8/v8.git/+/3.14.5.9/samples/shell.cc
void reportException(TryCatch* _tryCatch)
{
	HandleScope handle_scope;
	String::Utf8Value exception(_tryCatch->Exception());
	char const* exceptionString = toCString(exception);
	Handle<Message> message = _tryCatch->Message();

	// V8 didn't provide any extra information about this error; just
	// print the exception.
	if (message.IsEmpty())
		printf("%s\n", exceptionString);
	else
	{
		// Print (filename):(line number): (message).
		String::Utf8Value filename(message->GetScriptResourceName());
		char const* filenameString = toCString(filename);
		int linenum = message->GetLineNumber();
		printf("%s:%i: %s\n", filenameString, linenum, exceptionString);

		// Print line of source code.
		String::Utf8Value sourceline(message->GetSourceLine());
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

		String::Utf8Value stackTrace(_tryCatch->StackTrace());
		if (stackTrace.length() > 0)
		{
			char const* stackTraceString = toCString(stackTrace);
			printf("%s\n", stackTraceString);
		}
	}
}

Handle<Value> ConsoleLog(Arguments const& _args)
{
	Local<External> wrap = Local<External>::Cast(_args.Data());
	auto engine = reinterpret_cast<JSV8Engine const*>(wrap->Value());
	JSV8Printer printer(*engine);
	for (int i = 0; i < _args.Length(); ++i)
		printf("%s\n", printer.prettyPrint(_args[i]).cstr());
	return Undefined();
}


class JSV8Scope
{
public:
	JSV8Scope():
		m_handleScope(),
		m_context(Context::New(NULL, ObjectTemplate::New())),
		m_contextScope(m_context)
	{
		m_context->Enter();
	}

	~JSV8Scope()
	{
		m_context->Exit();
		m_context.Dispose();
	}

	Persistent <Context> const& context() const { return m_context; }

private:
	HandleScope m_handleScope;
	Persistent <Context> m_context;
	Context::Scope m_contextScope;
};

}
}

JSString JSV8Value::toString() const
{
	if (m_value.IsEmpty())
		return "";

	else if (m_value->IsUndefined())
		return "undefined";

	String::Utf8Value str(m_value);
	return toCString(str);
}

JSV8Engine::JSV8Engine(): m_scope(new JSV8Scope())
{
	JSEngineResources resources;
	eval("env = typeof(env) === 'undefined' ? {} : env; env.os = '" TO_STRING(ETH_BUILD_PLATFORM) "'");
	string common = resources.loadResourceAsString("common");
	string web3 = resources.loadResourceAsString("web3");
	string admin = resources.loadResourceAsString("admin");
	eval(common.c_str());
	eval(web3.c_str());
	eval("web3 = require('web3');");
	eval(admin.c_str());

	auto consoleTemplate = ObjectTemplate::New();

	Local<FunctionTemplate> function = FunctionTemplate::New(ConsoleLog, External::New(this));
	consoleTemplate->Set(String::New("debug"), function);
	consoleTemplate->Set(String::New("log"), function);
	consoleTemplate->Set(String::New("error"), function);
	context()->Global()->Set(String::New("console"), consoleTemplate->NewInstance());
}

JSV8Engine::~JSV8Engine()
{
	delete m_scope;
}

JSV8Value JSV8Engine::eval(char const* _cstr) const
{
	TryCatch tryCatch;
	Local<String> source = String::New(_cstr);
	Local<String> name(String::New("(shell)"));
	ScriptOrigin origin(name);
	Handle<Script> script = Script::Compile(source, &origin);

	// Make sure to wrap the exception in a new handle because
	// the handle returned from the TryCatch is destroyed
	if (script.IsEmpty())
	{
		reportException(&tryCatch);
		return Exception::Error(Local<String>::New(tryCatch.Message()->Get()));
	}

	auto result = script->Run();

	if (result.IsEmpty())
	{
		reportException(&tryCatch);
		return Exception::Error(Local<String>::New(tryCatch.Message()->Get()));
	}

	return result;
}

Handle<Context> const& JSV8Engine::context() const
{
	return m_scope->context();
}
