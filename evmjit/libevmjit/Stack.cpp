#include "Stack.h"
#include "RuntimeManager.h"
#include "Runtime.h"

#include <llvm/IR/Function.h>

namespace dev
{
namespace eth
{
namespace jit
{

Stack::Stack(llvm::IRBuilder<>& _builder, RuntimeManager& _runtimeManager):
	CompilerHelper(_builder),
	m_runtimeManager(_runtimeManager)
{
	m_arg = m_builder.CreateAlloca(Type::Word, nullptr, "stack.arg");

	using namespace llvm;
	using Linkage = GlobalValue::LinkageTypes;

	auto module = getModule();

	llvm::Type* pushArgTypes[] = {Type::RuntimePtr, Type::WordPtr};
	m_push = Function::Create(FunctionType::get(Type::Void, pushArgTypes, false), Linkage::ExternalLinkage, "stack_push", module);

	llvm::Type* getSetArgTypes[] = {Type::RuntimePtr, Type::Size, Type::WordPtr};
	m_set = Function::Create(FunctionType::get(Type::Void, getSetArgTypes, false), Linkage::ExternalLinkage, "stack_set", module);
}

llvm::Function* Stack::getPopFunc()
{
	auto& func = m_pop;
	if (!func)
	{
		llvm::Type* argTypes[] = {Type::RuntimePtr, Type::Size};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::ExternalLinkage, "stack.pop", getModule());
		auto extPopFunc = llvm::Function::Create(llvm::FunctionType::get(Type::Bool, argTypes, false), llvm::Function::ExternalLinkage, "stack_pop", getModule());

		auto rt = &func->getArgumentList().front();
		rt->setName("rt");
		auto index = rt->getNextNode();
		index->setName("index");

		InsertPointGuard guard{m_builder};
		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, func);
		auto underflowBB = llvm::BasicBlock::Create(m_builder.getContext(), "Underflow", func);
		auto returnBB = llvm::BasicBlock::Create(m_builder.getContext(), "Return", func);

		m_builder.SetInsertPoint(entryBB);
		auto ok = createCall(extPopFunc, {rt, index});
		m_builder.CreateCondBr(ok, returnBB, underflowBB);

		m_builder.SetInsertPoint(underflowBB);
		m_runtimeManager.raiseException(ReturnCode::StackTooSmall);
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
		llvm::Type* argTypes[] = {Type::RuntimePtr, Type::Size};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::WordPtr, argTypes, false), llvm::Function::ExternalLinkage, "stack.get", getModule());
		auto extGetFunc = llvm::Function::Create(llvm::FunctionType::get(Type::WordPtr, argTypes, false), llvm::Function::ExternalLinkage, "stack_get", getModule());

		auto rt = &func->getArgumentList().front();
		rt->setName("rt");
		auto index = rt->getNextNode();
		index->setName("index");

		InsertPointGuard guard{m_builder};
		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, func);
		auto underflowBB = llvm::BasicBlock::Create(m_builder.getContext(), "Underflow", func);
		auto returnBB = llvm::BasicBlock::Create(m_builder.getContext(), "Return", func);

		m_builder.SetInsertPoint(entryBB);
		auto valuePtr = createCall(extGetFunc, {rt, index});
		auto ok = m_builder.CreateICmpNE(valuePtr, llvm::ConstantPointerNull::get(Type::WordPtr));
		m_builder.CreateCondBr(ok, returnBB, underflowBB);

		m_builder.SetInsertPoint(underflowBB);
		m_runtimeManager.raiseException(ReturnCode::StackTooSmall);
		m_builder.CreateUnreachable();

		m_builder.SetInsertPoint(returnBB);
		m_builder.CreateRet(valuePtr);
	}
	return func;
}

llvm::Value* Stack::get(size_t _index)
{
	auto valuePtr = createCall(getGetFunc(), {m_runtimeManager.getRuntimePtr(), m_builder.getInt64(_index)});
	return m_builder.CreateLoad(valuePtr);
}

void Stack::set(size_t _index, llvm::Value* _value)
{
	m_builder.CreateStore(_value, m_arg);
	m_builder.CreateCall3(m_set, m_runtimeManager.getRuntimePtr(), llvm::ConstantInt::get(Type::Size, _index, false), m_arg);
}

void Stack::pop(size_t _count)
{
	createCall(getPopFunc(), {m_runtimeManager.getRuntimePtr(), m_builder.getInt64(_count)});
}

void Stack::push(llvm::Value* _value)
{
	m_builder.CreateStore(_value, m_arg);
	m_builder.CreateCall2(m_push, m_runtimeManager.getRuntimePtr(), m_arg);
}


size_t Stack::maxStackSize = 0;

}
}
}

extern "C"
{
	using namespace dev::eth::jit;

	EXPORT bool stack_pop(Runtime* _rt, uint64_t _count)
	{
		auto& stack = _rt->getStack();
		if (stack.size() < _count)
			return false;

		stack.erase(stack.end() - _count, stack.end());
		return true;
	}

	EXPORT void stack_push(Runtime* _rt, i256 const* _word)
	{
		auto& stack = _rt->getStack();
		stack.push_back(*_word);

		if (stack.size() > Stack::maxStackSize)
			Stack::maxStackSize = stack.size();
	}

	EXPORT i256* stack_get(Runtime* _rt, uint64_t _index)
	{
		auto& stack = _rt->getStack();
		return _index < stack.size() ? &*(stack.rbegin() + _index) : nullptr;
	}

	EXPORT void stack_set(Runtime* _rt, uint64_t _index, i256 const* _word)
	{
		auto& stack = _rt->getStack();
		assert(_index < stack.size());
		if (_index >= stack.size())
			return;

		*(stack.rbegin() + _index) = *_word;
	}

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

