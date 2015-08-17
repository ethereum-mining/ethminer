#include "Stack.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Function.h>
#include "preprocessor/llvm_includes_end.h"

#include "RuntimeManager.h"
#include "Utils.h"

namespace dev
{
namespace eth
{
namespace jit
{

Stack::Stack(llvm::IRBuilder<>& _builder):
	CompilerHelper(_builder),
	m_stack(_builder, "stack")
{}

llvm::Value* Stack::get(size_t _index)
{
	return m_stack.get(m_builder.CreateSub(m_stack.size(), m_builder.getInt64(_index + 1)));
}

void Stack::set(size_t _index, llvm::Value* _value)
{
	m_stack.set(m_builder.CreateSub(m_stack.size(), m_builder.getInt64(_index + 1)), _value);
}

void Stack::pop(size_t _count)
{
	m_stack.pop(m_builder.getInt64(_count));
}

void Stack::push(llvm::Value* _value)
{
	m_stack.push(_value);
}

}
}
}
