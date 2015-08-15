#pragma once

#include <functional>

#include "CompilerHelper.h"

namespace dev
{
namespace eth
{
namespace jit
{

class LazyFunction
{
public:
	using Creator = std::function<llvm::Function*()>;

	LazyFunction(Creator _creator) :
		m_creator(_creator)
	{}

	llvm::Value* call(llvm::IRBuilder<>& _builder, std::initializer_list<llvm::Value*> const& _args, llvm::Twine const& _name = "");

private:
	llvm::Function* m_func = nullptr;
	Creator m_creator;
};

class Array : public CompilerHelper
{
public:
	Array(llvm::IRBuilder<>& _builder, char const* _name);
	Array(llvm::IRBuilder<>& _builder, llvm::Value* _array);

	void push(llvm::Value* _value) { m_pushFunc.call(m_builder, {m_array, _value}); }
	void set(llvm::Value* _index, llvm::Value* _value) { m_setFunc.call(m_builder, {m_array, _index, _value}); }
	llvm::Value* get(llvm::Value* _index) { return m_getFunc.call(m_builder, {m_array, _index}); }
	void pop(llvm::Value* _count);
	llvm::Value* size(llvm::Value* _array = nullptr);
	void free() { m_freeFunc.call(m_builder, {m_array}); }

	void extend(llvm::Value* _arrayPtr, llvm::Value* _size);
	llvm::Value* getPtr(llvm::Value* _arrayPtr, llvm::Value* _index) { return m_getPtrFunc.call(m_builder, {_arrayPtr, _index}); }

	llvm::Value* getPointerTo() const { return m_array; }

	static llvm::Type* getType();

private:
	llvm::Value* m_array = nullptr;

	llvm::Function* createArrayPushFunc();
	llvm::Function* createArraySetFunc();
	llvm::Function* createArrayGetFunc();
	llvm::Function* createGetPtrFunc();
	llvm::Function* createFreeFunc();
	llvm::Function* createExtendFunc();
	llvm::Function* getReallocFunc();

	LazyFunction m_pushFunc = {[this](){ return createArrayPushFunc(); }};
	LazyFunction m_setFunc = {[this](){ return createArraySetFunc(); }};
	LazyFunction m_getPtrFunc = {[this](){ return createGetPtrFunc(); }};
	LazyFunction m_getFunc = {[this](){ return createArrayGetFunc(); }};
	LazyFunction m_freeFunc = {[this](){ return createFreeFunc(); }};
	LazyFunction m_extendFunc = {[this](){ return createExtendFunc(); }};
	LazyFunction m_reallocFunc = {[this](){ return getReallocFunc(); }};
};

}
}
}
