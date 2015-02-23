#include "Stack.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Function.h>
#include "preprocessor/llvm_includes_end.h"

#include "RuntimeManager.h"
#include "Runtime.h"

#include <iostream>
#include <set>

namespace dev
{
namespace eth
{
namespace jit
{

static const auto c_reallocStep = 1;
static const auto c_reallocMultipier = 2;

llvm::Value* LazyFunction::call(llvm::IRBuilder<>& _builder, std::initializer_list<llvm::Value*> const& _args)
{
	if (!m_func)
		m_func = m_creator();
	
	return _builder.CreateCall(m_func, {_args.begin(), _args.size()});
}

llvm::Function* Array::createArrayPushFunc()
{
	llvm::Type* argTypes[] = {m_array->getType(), Type::Word};
	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::PrivateLinkage, "array.push", getModule());
	func->setDoesNotThrow();
	func->setDoesNotCapture(1);

	llvm::Type* reallocArgTypes[] = {Type::BytePtr, Type::Size};
	auto reallocFunc = llvm::Function::Create(llvm::FunctionType::get(Type::BytePtr, reallocArgTypes, false), llvm::Function::ExternalLinkage, "ext_realloc", getModule());
	reallocFunc->setDoesNotThrow();
	reallocFunc->setDoesNotAlias(0);
	reallocFunc->setDoesNotCapture(1);

	auto arrayPtr = &func->getArgumentList().front();
	arrayPtr->setName("arrayPtr");
	auto value = arrayPtr->getNextNode();
	value->setName("value");

	InsertPointGuard guard{m_builder};
	auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), "Entry", func);
	auto reallocBB = llvm::BasicBlock::Create(m_builder.getContext(), "Realloc", func);
	auto pushBB = llvm::BasicBlock::Create(m_builder.getContext(), "Push", func);

	m_builder.SetInsertPoint(entryBB);
	auto dataPtr = m_builder.CreateStructGEP(arrayPtr, 0, "dataPtr");
	auto sizePtr = m_builder.CreateStructGEP(arrayPtr, 1, "sizePtr");
	auto capPtr = m_builder.CreateStructGEP(arrayPtr, 2, "capPtr");
	auto data = m_builder.CreateLoad(dataPtr, "data");
	auto size = m_builder.CreateLoad(sizePtr, "size");
	auto cap = m_builder.CreateLoad(capPtr, "cap");
	auto reallocReq = m_builder.CreateICmpEQ(cap, size, "reallocReq");
	m_builder.CreateCondBr(reallocReq, reallocBB, pushBB);

	m_builder.SetInsertPoint(reallocBB);
	auto newCap = m_builder.CreateNUWAdd(cap, m_builder.getInt64(c_reallocStep), "newCap");
	//newCap = m_builder.CreateNUWMul(newCap, m_builder.getInt64(c_reallocMultipier));
	auto reallocSize = m_builder.CreateShl(newCap, 5, "reallocSize"); // size in bytes: newCap * 32
	auto bytes = m_builder.CreateBitCast(data, Type::BytePtr, "bytes");
	auto newBytes = m_builder.CreateCall2(reallocFunc, bytes, reallocSize, "newBytes");
	auto newData = m_builder.CreateBitCast(newBytes, Type::WordPtr, "newData");
	m_builder.CreateStore(newData, dataPtr);
	m_builder.CreateStore(newCap, capPtr);
	m_builder.CreateBr(pushBB);
	
	m_builder.SetInsertPoint(pushBB);
	auto dataPhi = m_builder.CreatePHI(Type::WordPtr, 2, "dataPhi");
	dataPhi->addIncoming(data, entryBB);
	dataPhi->addIncoming(newData, reallocBB);
	auto newElemPtr = m_builder.CreateGEP(dataPhi, size, "newElemPtr");
	m_builder.CreateStore(value, newElemPtr);
	auto newSize = m_builder.CreateNUWAdd(size, m_builder.getInt64(1), "newSize");
	m_builder.CreateStore(newSize, sizePtr);
	m_builder.CreateRetVoid();

	return func;
}

llvm::Function* Array::createArraySetFunc()
{
	llvm::Type* argTypes[] = {m_array->getType(), Type::Size, Type::Word};
	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::PrivateLinkage, "array.set", getModule());
	func->setDoesNotThrow();
	func->setDoesNotCapture(1);

	auto arrayPtr = &func->getArgumentList().front();
	arrayPtr->setName("arrayPtr");
	auto index = arrayPtr->getNextNode();
	index->setName("index");
	auto value = index->getNextNode();
	value->setName("value");

	InsertPointGuard guard{m_builder};
	m_builder.SetInsertPoint(llvm::BasicBlock::Create(m_builder.getContext(), {}, func));
	auto dataPtr = m_builder.CreateStructGEP(arrayPtr, 0, "dataPtr");
	auto data = m_builder.CreateLoad(dataPtr, "data");
	auto valuePtr = m_builder.CreateGEP(data, index, "valuePtr");
	m_builder.CreateStore(value, valuePtr);
	m_builder.CreateRetVoid();
	return func;
}

llvm::Function* Array::createArrayGetFunc()
{
	llvm::Type* argTypes[] = {m_array->getType(), Type::Size};
	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Word, argTypes, false), llvm::Function::PrivateLinkage, "array.get", getModule());
	func->setDoesNotThrow();
	func->setDoesNotCapture(1);

	auto arrayPtr = &func->getArgumentList().front();
	arrayPtr->setName("arrayPtr");
	auto index = arrayPtr->getNextNode();
	index->setName("index");

	InsertPointGuard guard{m_builder};
	m_builder.SetInsertPoint(llvm::BasicBlock::Create(m_builder.getContext(), {}, func));
	auto dataPtr = m_builder.CreateStructGEP(arrayPtr, 0, "dataPtr");
	auto data = m_builder.CreateLoad(dataPtr, "data");
	auto valuePtr = m_builder.CreateGEP(data, index, "valuePtr");
	auto value = m_builder.CreateLoad(valuePtr, "value");
	m_builder.CreateRet(value);
	return func;
}

llvm::Function* Array::createFreeFunc()
{
	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, m_array->getType(), false), llvm::Function::PrivateLinkage, "array.free", getModule());
	func->setDoesNotThrow();
	func->setDoesNotCapture(1);

	auto freeFunc = llvm::Function::Create(llvm::FunctionType::get(Type::Void, Type::BytePtr, false), llvm::Function::ExternalLinkage, "ext_free", getModule());
	freeFunc->setDoesNotThrow();
	freeFunc->setDoesNotCapture(1);

	auto arrayPtr = &func->getArgumentList().front();
	arrayPtr->setName("arrayPtr");

	InsertPointGuard guard{m_builder};
	m_builder.SetInsertPoint(llvm::BasicBlock::Create(m_builder.getContext(), {}, func));
	auto dataPtr = m_builder.CreateStructGEP(arrayPtr, 0, "dataPtr");
	auto data = m_builder.CreateLoad(dataPtr, "data");
	auto mem = m_builder.CreateBitCast(data, Type::BytePtr, "mem");
	m_builder.CreateCall(freeFunc, mem);
	m_builder.CreateRetVoid();
	return func;
}

Array::Array(llvm::IRBuilder<>& _builder, char const* _name) :
	CompilerHelper(_builder),
	m_pushFunc([this](){ return createArrayPushFunc(); }),
	m_setFunc([this](){ return createArraySetFunc(); }),
	m_getFunc([this](){ return createArrayGetFunc(); }),
	m_freeFunc([this](){ return createFreeFunc(); })
{
	llvm::Type* elementTys[] = {Type::WordPtr, Type::Size, Type::Size};
	static auto arrayTy = llvm::StructType::create(elementTys, "Array");

	m_array = m_builder.CreateAlloca(arrayTy, nullptr, _name);
	m_builder.CreateStore(llvm::ConstantAggregateZero::get(arrayTy), m_array);
}

void Array::pop(llvm::Value* _count)
{
	auto sizePtr = m_builder.CreateStructGEP(m_array, 1, "sizePtr");
	auto size = m_builder.CreateLoad(sizePtr, "size");
	auto newSize = m_builder.CreateNUWSub(size, _count, "newSize");
	m_builder.CreateStore(newSize, sizePtr);
}

llvm::Value* Array::size()
{
	auto sizePtr = m_builder.CreateStructGEP(m_array, 1, "sizePtr");
	return m_builder.CreateLoad(sizePtr, "array.size");
}

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
	//createCall(getSetFunc(), {m_runtimeManager.getRuntimePtr(), m_builder.getInt64(_index), _value});
}

void Stack::pop(size_t _count)
{
	// FIXME: Pop does not check for stack underflow but looks like not needed
	//        We should place stack.require() check and begining of every BB
	m_stack.pop(m_builder.getInt64(_count));
	//createCall(getPopFunc(), {m_runtimeManager.getRuntimePtr(), m_builder.getInt64(_count), m_runtimeManager.getJmpBuf()});
}

void Stack::push(llvm::Value* _value)
{
	m_stack.push(_value);
	//createCall(getPushFunc(), {m_runtimeManager.getRuntimePtr(), _value});
}

}
}
}

namespace
{
	struct AllocatedMemoryWatchdog
	{
		std::set<void*> allocatedMemory;

		~AllocatedMemoryWatchdog()
		{
			if (!allocatedMemory.empty())
				std::cerr << allocatedMemory.size() << " MEM LEAKS!" << std::endl;
		}
	};

	AllocatedMemoryWatchdog watchdog;
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

	EXPORT void* ext_realloc(void* _data, size_t _size)
	{
		//std::cerr << "REALLOC: " << _data << " [" << _size << "]" << std::endl;
		auto newData = std::realloc(_data, _size);
		if (_data != newData)
		{
			std::cerr << "REALLOC: " << _data << " -> " << newData << " [" << _size << "]" << std::endl;
			watchdog.allocatedMemory.erase(_data);
			watchdog.allocatedMemory.insert(newData);
		}
		return newData;
	}

	EXPORT void ext_free(void* _data)
	{
		std::free(_data);
		if (_data)
		{
			std::cerr << "FREE   : " << _data << std::endl;
			watchdog.allocatedMemory.erase(_data);
		}
	}

} // extern "C"

