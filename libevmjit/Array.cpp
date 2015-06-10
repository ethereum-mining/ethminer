#include "Array.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include "preprocessor/llvm_includes_end.h"

#include "RuntimeManager.h"
#include "Runtime.h"
#include "Utils.h"

namespace dev
{
namespace eth
{
namespace jit
{

static const auto c_reallocStep = 1;
static const auto c_reallocMultipier = 2;

llvm::Value* LazyFunction::call(llvm::IRBuilder<>& _builder, std::initializer_list<llvm::Value*> const& _args, llvm::Twine const& _name)
{
	if (!m_func)
		m_func = m_creator();

	return _builder.CreateCall(m_func, {_args.begin(), _args.size()}, _name);
}

llvm::Function* Array::createArrayPushFunc()
{
	llvm::Type* argTypes[] = {m_array->getType(), Type::Word};
	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::PrivateLinkage, "array.push", getModule());
	func->setDoesNotThrow();
	func->setDoesNotCapture(1);

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
	auto newBytes = m_reallocFunc.call(m_builder, {bytes, reallocSize}, "newBytes");
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

llvm::Function* Array::createGetPtrFunc()
{
	llvm::Type* argTypes[] = {m_array->getType(), Type::Size};
	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::WordPtr, argTypes, false), llvm::Function::PrivateLinkage, "array.getPtr", getModule());
	func->setDoesNotThrow();
	func->setDoesNotCapture(1);

	auto arrayPtr = &func->getArgumentList().front();
	arrayPtr->setName("arrayPtr");
	auto index = arrayPtr->getNextNode();
	index->setName("index");

	InsertPointGuard guard{m_builder};
	m_builder.SetInsertPoint(llvm::BasicBlock::Create(m_builder.getContext(), {}, func));
	auto dataPtr = m_builder.CreateBitCast(arrayPtr, Type::BytePtr->getPointerTo(), "dataPtr");
	auto data = m_builder.CreateLoad(dataPtr, "data");
	auto bytePtr = m_builder.CreateGEP(data, index, "bytePtr");
	auto wordPtr = m_builder.CreateBitCast(bytePtr, Type::WordPtr, "wordPtr");
	m_builder.CreateRet(wordPtr);
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

llvm::Function* Array::getReallocFunc()
{
	if (auto func = getModule()->getFunction("ext_realloc"))
		return func;

	llvm::Type* reallocArgTypes[] = {Type::BytePtr, Type::Size};
	auto reallocFunc = llvm::Function::Create(llvm::FunctionType::get(Type::BytePtr, reallocArgTypes, false), llvm::Function::ExternalLinkage, "ext_realloc", getModule());
	reallocFunc->setDoesNotThrow();
	reallocFunc->setDoesNotAlias(0);
	reallocFunc->setDoesNotCapture(1);
	return reallocFunc;
}

llvm::Function* Array::createExtendFunc()
{
	llvm::Type* argTypes[] = {m_array->getType(), Type::Size};
	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::PrivateLinkage, "array.extend", getModule());
	func->setDoesNotThrow();
	func->setDoesNotCapture(1);

	auto arrayPtr = &func->getArgumentList().front();
	arrayPtr->setName("arrayPtr");
	auto newSize = arrayPtr->getNextNode();
	newSize->setName("newSize");

	InsertPointGuard guard{m_builder};
	m_builder.SetInsertPoint(llvm::BasicBlock::Create(m_builder.getContext(), {}, func));
	auto dataPtr = m_builder.CreateBitCast(arrayPtr, Type::BytePtr->getPointerTo(), "dataPtr");// TODO: Use byte* in Array
	auto sizePtr = m_builder.CreateStructGEP(arrayPtr, 1, "sizePtr");
	auto capPtr = m_builder.CreateStructGEP(arrayPtr, 2, "capPtr");
	auto data = m_builder.CreateLoad(dataPtr, "data");
	auto size = m_builder.CreateLoad(sizePtr, "size");
	auto extSize = m_builder.CreateNUWSub(newSize, size, "extSize");
	auto newData = m_reallocFunc.call(m_builder, {data, newSize}, "newData"); // TODO: Check realloc result for null
	auto extPtr = m_builder.CreateGEP(newData, size, "extPtr");
	m_builder.CreateMemSet(extPtr, m_builder.getInt8(0), extSize, 16);
	m_builder.CreateStore(newData, dataPtr);
	m_builder.CreateStore(newSize, sizePtr);
	m_builder.CreateStore(newSize, capPtr);
	m_builder.CreateRetVoid();
	return func;
}

llvm::Type* Array::getType()
{
	llvm::Type* elementTys[] = {Type::WordPtr, Type::Size, Type::Size};
	static auto arrayTy = llvm::StructType::create(elementTys, "Array");
	return arrayTy;
}

Array::Array(llvm::IRBuilder<>& _builder, char const* _name) :
	CompilerHelper(_builder),
	m_pushFunc([this](){ return createArrayPushFunc(); }),
	m_setFunc([this](){ return createArraySetFunc(); }),
	m_getFunc([this](){ return createArrayGetFunc(); }),
	m_freeFunc([this](){ return createFreeFunc(); })
{
	m_array = m_builder.CreateAlloca(getType(), nullptr, _name);
	m_builder.CreateStore(llvm::ConstantAggregateZero::get(getType()), m_array);
}

Array::Array(llvm::IRBuilder<>& _builder, llvm::Value* _array) :
	CompilerHelper(_builder),
	m_array(_array),
	m_pushFunc([this](){ return createArrayPushFunc(); }),
	m_setFunc([this](){ return createArraySetFunc(); }),
	m_getFunc([this](){ return createArrayGetFunc(); }),
	m_freeFunc([this](){ return createFreeFunc(); })
{
	m_builder.CreateStore(llvm::ConstantAggregateZero::get(getType()), m_array);
}


void Array::pop(llvm::Value* _count)
{
	auto sizePtr = m_builder.CreateStructGEP(m_array, 1, "sizePtr");
	auto size = m_builder.CreateLoad(sizePtr, "size");
	auto newSize = m_builder.CreateNUWSub(size, _count, "newSize");
	m_builder.CreateStore(newSize, sizePtr);
}

llvm::Value* Array::size(llvm::Value* _array)
{
	auto sizePtr = m_builder.CreateStructGEP(_array ? _array : m_array, 1, "sizePtr");
	return m_builder.CreateLoad(sizePtr, "array.size");
}

void Array::extend(llvm::Value* _arrayPtr, llvm::Value* _size)
{
	assert(_arrayPtr->getType() == m_array->getType());
	assert(_size->getType() == Type::Size);
	m_extendFunc.call(m_builder, {_arrayPtr, _size});
}

}
}
}

extern "C"
{
	EXPORT void* ext_realloc(void* _data, size_t _size) noexcept
	{
		return std::realloc(_data, _size);
	}

	EXPORT void ext_free(void* _data) noexcept
	{
		std::free(_data);
	}
}
