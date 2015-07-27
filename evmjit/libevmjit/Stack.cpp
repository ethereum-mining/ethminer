#include "Stack.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Function.h>
#include "preprocessor/llvm_includes_end.h"

#include "RuntimeManager.h"
#include "Utils.h"

#include <set> // DEBUG only

namespace dev
{
namespace eth
{
namespace jit
{

Stack::Stack(llvm::IRBuilder<>& _builder, RuntimeManager& _runtimeManager):
	CompilerHelper(_builder),
	m_runtimeManager(_runtimeManager),
	m_stack(_builder, "stack")
{}

llvm::Function* Stack::getGetFunc()
{
	auto& func = m_get;
	if (!func)
	{
		llvm::Type* argTypes[] = {Type::Size, Type::Size, Type::BytePtr};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::ExternalLinkage, "stack.require", getModule());

		auto index = &func->getArgumentList().front();
		index->setName("index");
		auto size = index->getNextNode();
		size->setName("size");
		auto jmpBuf = size->getNextNode();
		jmpBuf->setName("jmpBuf");

		InsertPointGuard guard{m_builder};
		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, func);
		auto underflowBB = llvm::BasicBlock::Create(m_builder.getContext(), "Underflow", func);
		auto returnBB = llvm::BasicBlock::Create(m_builder.getContext(), "Return", func);

		m_builder.SetInsertPoint(entryBB);
		auto underflow = m_builder.CreateICmpUGE(index, size, "underflow");
		m_builder.CreateCondBr(underflow, underflowBB, returnBB);

		m_builder.SetInsertPoint(underflowBB);
		m_runtimeManager.abort(jmpBuf);
		m_builder.CreateUnreachable();

		m_builder.SetInsertPoint(returnBB);
		m_builder.CreateRetVoid();
	}
	return func;
}

llvm::Value* Stack::get(size_t _index)
{
	createCall(getGetFunc(), {m_builder.getInt64(_index), m_stack.size(), m_runtimeManager.getJmpBuf()});
	auto value = m_stack.get(m_builder.CreateSub(m_stack.size(), m_builder.getInt64(_index + 1)));
	//return m_builder.CreateLoad(valuePtr);
	return value;
}

void Stack::set(size_t _index, llvm::Value* _value)
{
	m_stack.set(m_builder.CreateSub(m_stack.size(), m_builder.getInt64(_index + 1)), _value);
}

void Stack::pop(size_t _count)
{
	// TODO: We should place stack.require() check and begining of every BB
	m_stack.pop(m_builder.getInt64(_count));
}

void Stack::push(llvm::Value* _value)
{
	m_stack.push(_value);
}

}
}
}
