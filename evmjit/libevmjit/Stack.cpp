#include "Stack.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Function.h>
#include "preprocessor/llvm_includes_end.h"

#include "RuntimeManager.h"
#include "Runtime.h"

namespace dev
{
namespace eth
{
namespace jit
{

Stack::Stack(llvm::IRBuilder<>& _builder, RuntimeManager& _runtimeManager):
	CompilerHelper(_builder),
	m_runtimeManager(_runtimeManager)
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
		auto extPushFunc = llvm::Function::Create(llvm::FunctionType::get(Type::Void, extArgTypes, false), llvm::Function::ExternalLinkage, "stack_set", getModule());

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
		createCall(extPushFunc, {rt, index, a});
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
		llvm::Type* argTypes[] = {Type::RuntimePtr, Type::Size, Type::BytePtr};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::WordPtr, argTypes, false), llvm::Function::ExternalLinkage, "stack.get", getModule());
		llvm::Type* extArgTypes[] = {Type::RuntimePtr, Type::Size};
		auto extGetFunc = llvm::Function::Create(llvm::FunctionType::get(Type::WordPtr, extArgTypes, false), llvm::Function::ExternalLinkage, "stack_get", getModule());

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
		auto valuePtr = createCall(extGetFunc, {rt, index});
		auto ok = m_builder.CreateICmpNE(valuePtr, llvm::ConstantPointerNull::get(Type::WordPtr));
		m_builder.CreateCondBr(ok, returnBB, underflowBB); //TODO: Add branch weight

		m_builder.SetInsertPoint(underflowBB);
		m_runtimeManager.abort(jmpBuf);
		m_builder.CreateUnreachable();

		m_builder.SetInsertPoint(returnBB);
		m_builder.CreateRet(valuePtr);
	}
	return func;
}

llvm::Value* Stack::get(size_t _index)
{
	auto valuePtr = createCall(getGetFunc(), {m_runtimeManager.getRuntimePtr(), m_builder.getInt64(_index), m_runtimeManager.getJmpBuf()});
	return m_builder.CreateLoad(valuePtr);
}

void Stack::set(size_t _index, llvm::Value* _value)
{
	createCall(getSetFunc(), {m_runtimeManager.getRuntimePtr(), m_builder.getInt64(_index), _value});
}

void Stack::pop(size_t _count)
{
	createCall(getPopFunc(), {m_runtimeManager.getRuntimePtr(), m_builder.getInt64(_count), m_runtimeManager.getJmpBuf()});
}

void Stack::push(llvm::Value* _value)
{
	createCall(getPushFunc(), {m_runtimeManager.getRuntimePtr(), _value});
}

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

