#pragma once

#include <functional>

#include "Array.h"

namespace dev
{
namespace eth
{
namespace jit
{
class RuntimeManager;

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
	llvm::Function* getGetFunc();

	RuntimeManager& m_runtimeManager;
	llvm::Function* m_get = nullptr;
	Array m_stack;
};


}
}
}


