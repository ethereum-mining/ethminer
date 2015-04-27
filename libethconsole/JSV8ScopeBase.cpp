//
// Created by Marek Kotewicz on 27/04/15.
//

#include <libplatform/libplatform.h>
#include <memory>
#include "JSV8ScopeBase.h"

using namespace dev;
using namespace dev::eth;

JSV8Env JSV8ScopeBase::s_env = JSV8Env();

class ShellArrayBufferAllocator : public v8::ArrayBuffer::Allocator {
public:
	virtual void* Allocate(size_t length) {
		void* data = AllocateUninitialized(length);
		return data == NULL ? data : memset(data, 0, length);
	}
	virtual void* AllocateUninitialized(size_t length) { return malloc(length); }
	virtual void Free(void* data, size_t) { free(data); }
};

JSV8Env::JSV8Env()
{
	static bool initialized = false;
	if (initialized)
		return;
	initialized = true;
	v8::V8::InitializeICU();
	m_platform = v8::platform::CreateDefaultPlatform();
	v8::V8::InitializePlatform(m_platform);
	v8::V8::Initialize();
	ShellArrayBufferAllocator array_buffer_allocator;
	v8::V8::SetArrayBufferAllocator(&array_buffer_allocator);
}

JSV8Env::~JSV8Env()
{
	v8::V8::Dispose();
	v8::V8::ShutdownPlatform();
	delete m_platform;
}

JSV8ScopeBase::JSV8ScopeBase():
		m_isolate(v8::Isolate::New()),
		m_scope(new JSV8DumbScope(m_isolate))
{ }

JSV8ScopeBase::~JSV8ScopeBase()
{
	delete m_scope;
	m_isolate->Dispose();
}

const char* JSV8ScopeBase::evaluate(const char* _cstr) const
{
	v8::HandleScope handleScope(m_isolate);
//	v8::TryCatch tryCatch;
	v8::Local<v8::String> source = v8::String::NewFromUtf8(m_scope->context()->GetIsolate(), _cstr);
	v8::Local<v8::String> name(v8::String::NewFromUtf8(m_scope->context()->GetIsolate(), "(shell)"));
	v8::ScriptOrigin origin(name);
	v8::Handle<v8::Script> script = v8::Script::Compile(source, &origin);
	if (script.IsEmpty())
	{
		// TODO: handle exceptions
		return "";
	}

	v8::Handle<v8::Value> result = script->Run();
	return formatValue(result);
}

const char* JSV8ScopeBase::formatValue(v8::Handle<v8::Value> const &_value) const
{
	if (_value.IsEmpty())
	{
		// TODO: handle exceptions
		return "";
	}
	else if (_value->IsUndefined())
	{
		return "undefined";
	}
	v8::String::Utf8Value str(_value);
	return *str ? *str : "<string conversion failed>";
}
