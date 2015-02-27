#include "Stack.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Function.h>
#include "preprocessor/llvm_includes_end.h"

#include "RuntimeManager.h"
#include "Runtime.h"
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

llvm::Function* Stack::getPushFunc()
{
	auto& func = m_push;
	if (!func)
	{
		llvm::Type* argTypes[] = {Type::RuntimePtr, Type::Word};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::ExternalLinkage, "stack.push", getModule());
		llvm::Type* extArgTypes[] = {Type::RuntimePtr, Type::WordPtr};
		auto extPushFunc = llvm::Function::Create(llvm::FunctionType::get(Type::Void, extArgTypes, false), llvm::Function::ExternalLinkage, "stack_push", getModule());

		auto rt = &func->getArgumentList().front();
		rt->setName("rt");
		auto value = rt->getNextNode();
		value->setName("value");

		InsertPointGuard guard{m_builder};
		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, func);
		m_builder.SetInsertPoint(entryBB);
		auto a = m_builder.CreateAlloca(Type::Word);
		m_builder.CreateStore(value, a);
		createCall(extPushFunc, {rt, a});
		m_builder.CreateRetVoid();
	}
	return func;
}

llvm::Function* Stack::getSetFunc()
{
	auto& func = m_set;
	if (!func)
	{
		llvm::Type* argTypes[] = {Type::RuntimePtr, Type::Size, Type::Word};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::ExternalLinkage, "stack.set", getModule());
		llvm::Type* extArgTypes[] = {Type::RuntimePtr, Type::Size, Type::WordPtr};
		auto extSetFunc = llvm::Function::Create(llvm::FunctionType::get(Type::Void, extArgTypes, false), llvm::Function::ExternalLinkage, "stack_set", getModule());

		auto rt = &func->getArgumentList().front();
		rt->setName("rt");
		auto index = rt->getNextNode();
		index->setName("index");
		auto value = index->getNextNode();
		value->setName("value");

		InsertPointGuard guard{m_builder};
		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, func);
		m_builder.SetInsertPoint(entryBB);
		auto a = m_builder.CreateAlloca(Type::Word);
		m_builder.CreateStore(value, a);
		createCall(extSetFunc, {rt, index, a});
		m_builder.CreateRetVoid();
	}
	return func;
}

llvm::Function* Stack::getPopFunc()
{
	auto& func = m_pop;
	if (!func)
	{
		llvm::Type* argTypes[] = {Type::RuntimePtr, Type::Size, Type::BytePtr};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::ExternalLinkage, "stack.pop", getModule());
		llvm::Type* extArgTypes[] = {Type::RuntimePtr, Type::Size};
		auto extPopFunc = llvm::Function::Create(llvm::FunctionType::get(Type::Bool, extArgTypes, false), llvm::Function::ExternalLinkage, "stack_pop", getModule());

		auto rt = &func->getArgumentList().front();
		rt->setName("rt");
		auto index = rt->getNextNode();
		index->setName("index");
		auto jmpBuf = index->getNextNode();
		jmpBuf->setName("jmpBuf");

		InsertPointGuard guard{m_builder};
		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, func);
		auto underflowBB = llvm::BasicBlock::Create(m_builder.getContext(), "Underflow", func);
		auto returnBB = llvm::BasicBlock::Create(m_builder.getContext(), "Return", func);

		m_builder.SetInsertPoint(entryBB);
		auto ok = createCall(extPopFunc, {rt, index});
		m_builder.CreateCondBr(ok, returnBB, underflowBB); //TODO: Add branch weight

		m_builder.SetInsertPoint(underflowBB);
		m_runtimeManager.abort(jmpBuf);
		m_builder.CreateUnreachable();

		m_builder.SetInsertPoint(returnBB);
		m_builder.CreateRetVoid();
	}
	return func;
}

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
	// FIXME: Pop does not check for stack underflow but looks like not needed
	//        We should place stack.require() check and begining of every BB
	m_stack.pop(m_builder.getInt64(_count));
}

void Stack::push(llvm::Value* _value)
{
	m_stack.push(_value);
}

}
}
}

extern "C"
{
	using namespace dev::eth::jit;

	EXPORT void ext_calldataload(RuntimeData* _rtData, i256* _index, byte* o_value)
	{
		// It asumes all indexes are less than 2^64

		auto index = _index->a;
		if (_index->b || _index->c || _index->d)				 // if bigger that 2^64
			index = std::numeric_limits<decltype(index)>::max(); // set max to fill with 0 leter

		auto data = _rtData->callData;
		auto size = _rtData->callDataSize;
		for (auto i = 0; i < 32; ++i)
		{
			if (index < size)
			{
				o_value[i] = data[index];
				++index;  // increment only if in range
			}
			else
				o_value[i] = 0;
		}
	}

} // extern "C"

