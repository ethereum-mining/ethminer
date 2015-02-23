#pragma once

#include <functional>

#include "CompilerHelper.h"

namespace dev
{
namespace eth
{
namespace jit
{
class RuntimeManager;

class LazyFunction
{
public:
	using Creator = std::function<llvm::Function*()>;

	LazyFunction(Creator _creator) :
		m_creator(_creator)
	{}

	llvm::Value* call(llvm::IRBuilder<>& _builder, std::initializer_list<llvm::Value*> const& _args);

private:
	llvm::Function* m_func = nullptr;
	Creator m_creator;
};

class Array : public CompilerHelper
{
public:
	Array(llvm::IRBuilder<>& _builder, char const* _name);

	void push(llvm::Value* _value) { m_pushFunc.call(m_builder, {m_array, _value}); }
	void set(llvm::Value* _index, llvm::Value* _value) { m_setFunc.call(m_builder, {m_array, _index, _value}); }
	llvm::Value* get(llvm::Value* _index) { return m_getFunc.call(m_builder, {m_array, _index}); }
	void pop(llvm::Value* _count);
	llvm::Value* size();
	void free() { m_freeFunc.call(m_builder, {m_array}); }

	llvm::Value* getPointerTo() const { return m_array; }

private:
	llvm::Value* m_array = nullptr;

	LazyFunction m_pushFunc;
	LazyFunction m_setFunc;
	LazyFunction m_getFunc;
	LazyFunction m_freeFunc;

	llvm::Function* createArrayPushFunc();
	llvm::Function* createArraySetFunc();
	llvm::Function* createArrayGetFunc();
	llvm::Function* createFreeFunc();
};

class Stack : public CompilerHelper
{
public:
	Stack(llvm::IRBuilder<>& builder, RuntimeManager& runtimeManager);

	llvm::Value* get(size_t _index);
	void set(size_t _index, llvm::Value* _value);
	void pop(size_t _count);
	void push(llvm::Value* _value);
	void free() { m_stack.free(); }

private:
	llvm::Function* getPopFunc();
	llvm::Function* getPushFunc();
	llvm::Function* getGetFunc();
	llvm::Function* getSetFunc();

	RuntimeManager& m_runtimeManager;

	llvm::Function* m_pop = nullptr;
	llvm::Function* m_push = nullptr;
	llvm::Function* m_get = nullptr;
	llvm::Function* m_set = nullptr;

	Array m_stack;
};


}
}
}


