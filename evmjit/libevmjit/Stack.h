#pragma once

#include "Array.h"

namespace dev
{
namespace eth
{
namespace jit
{

class Stack: public CompilerHelper
{
public:
	Stack(llvm::IRBuilder<>& builder);

	llvm::Value* get(size_t _index);
	void set(size_t _index, llvm::Value* _value);
	void pop(size_t _count);
	void push(llvm::Value* _value);
	void free() { m_stack.free(); }

private:
	Array m_stack;
};


}
}
}
