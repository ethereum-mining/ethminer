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
//#include <libplatform/libplatform.h>
#include "JSV8Engine.h"
#include "libjsengine/JSEngineResources.hpp"

using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace eth
{

static const char* toCString(v8::String::Utf8Value const& value)
{
	return *value ? *value : "<string conversion failed>";
}

// from https://github.com/v8/v8-git-mirror/blob/master/samples/shell.cc
static void reportException(v8::Isolate* isolate, v8::TryCatch* try_catch)
{
//	v8::HandleScope handle_scope;
//	v8::String::Utf8Value exception(try_catch->Exception());
//	const char* exception_string = toCString(exception);
//	v8::Handle<v8::Message> message = try_catch->Message();
//
//	 V8 didn't provide any extra information about this error; just
//	 print the exception.
//	if (message.IsEmpty())
//		fprintf(stderr, "%s\n", exception_string);
//	else
//	{
//		 Print (filename):(line number): (message).
//		v8::String::Utf8Value filename(message->GetScriptOrigin().ResourceName());
//		const char* filename_string = toCString(filename);
//		int linenum = message->GetLineNumber();
//		fprintf(stderr, "%s:%i: %s\n", filename_string, linenum, exception_string);
//
//		 Print line of source code.
//		v8::String::Utf8Value sourceline(message->GetSourceLine());
//		const char* sourceline_string = toCString(sourceline);
//		fprintf(stderr, "%s\n", sourceline_string);
//
//		 Print wavy underline (GetUnderline is deprecated).
//		int start = message->GetStartColumn();
//
//		for (int i = 0; i < start; i++)
//			fprintf(stderr, " ");
//
//		int end = message->GetEndColumn();
//
//		for (int i = start; i < end; i++)
//			fprintf(stderr, "^");
//
//		fprintf(stderr, "\n");
//		v8::String::Utf8Value stack_trace(try_catch->StackTrace());
//
//		if (stack_trace.length() > 0)
//		{
//			const char* stack_trace_string = toCString(stack_trace);
//			fprintf(stderr, "%s\n", stack_trace_string);
//		}
//	}
}

//class ShellArrayBufferAllocator : public v8::ArrayBuffer::Allocator {
//public:
//	virtual void* Allocate(size_t length) {
//		void* data = AllocateUninitialized(length);
//		return data == NULL ? data : memset(data, 0, length);
//	}
//	virtual void* AllocateUninitialized(size_t length) { return malloc(length); }
//	virtual void Free(void* data, size_t) { free(data); }
//};

class JSV8Env
{
public:
	JSV8Env();

	~JSV8Env();

private:
//	v8::Platform *m_platform;
};

v8::Handle<v8::Context> createShellContext()
{
	v8::Handle<v8::ObjectTemplate> global = v8::ObjectTemplate::New();
	return v8::Context::New(NULL, global);
}

class JSV8Scope
{
public:
	JSV8Scope():
			m_handleScope(),
			m_context(createShellContext()),
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

JSV8Env JSV8Engine::s_env = JSV8Env();

const char* JSV8Value::asCString() const
{
	if (m_value.IsEmpty())
		return "";

	else if (m_value->IsUndefined())
		return "undefined";

	v8::String::Utf8Value str(m_value);
	return toCString(str);
}

JSV8Env::JSV8Env()
{
//	v8::V8::InitializeICU();
//	m_platform = v8::platform::CreateDefaultPlatform();
//	v8::V8::InitializePlatform(m_platform);
//	v8::V8::Initialize();
//	ShellArrayBufferAllocator array_buffer_allocator;
//	v8::V8::SetArrayBufferAllocator(&array_buffer_allocator);
}

JSV8Env::~JSV8Env()
{
	v8::V8::Dispose();
//	v8::V8::ShutdownPlatform();
//	delete m_platform;
}

JSV8Engine::JSV8Engine(): m_scope(new JSV8Scope())
{
	JSEngineResources resources;
	string common = resources.loadResourceAsString("common");
	string web3 = resources.loadResourceAsString("web3");
	eval(common.c_str());
	eval(web3.c_str());
	eval("web3 = require('web3');");
}

JSV8Engine::~JSV8Engine()
{
	delete m_scope;
}

JSV8Value JSV8Engine::eval(const char* _cstr) const
{
	v8::HandleScope handleScope;
	v8::TryCatch tryCatch;
	v8::Local<v8::String> source = v8::String::New(_cstr);
	v8::Local<v8::String> name(v8::String::New("(shell)"));
	v8::ScriptOrigin origin(name);
	v8::Handle<v8::Script> script = v8::Script::Compile(source, &origin);

	// Make sure to wrap the exception in a new handle because
	// the handle returned from the TryCatch is destroyed
	if (script.IsEmpty())
	{
//		reportException(&tryCatch);
		return v8::Exception::Error(v8::Local<v8::String>::New(tryCatch.Message()->Get()));
	}

	auto result = script->Run();

	if (result.IsEmpty())
	{
//		reportException(&tryCatch);
		return v8::Exception::Error(v8::Local<v8::String>::New(tryCatch.Message()->Get()));
	}

	return result;
}

v8::Handle<v8::Context> const& JSV8Engine::context() const
{
	return m_scope->context();
}
