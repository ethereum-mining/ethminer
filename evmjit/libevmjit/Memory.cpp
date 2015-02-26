#include "Memory.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/IntrinsicInst.h>
#include "preprocessor/llvm_includes_end.h"

#include "Type.h"
#include "Runtime.h"
#include "GasMeter.h"
#include "Endianness.h"
#include "RuntimeManager.h"

namespace dev
{
namespace eth
{
namespace jit
{

Memory::Memory(RuntimeManager& _runtimeManager, GasMeter& _gasMeter):
	RuntimeHelper(_runtimeManager),  // TODO: RuntimeHelper not needed
	m_memory{getBuilder(), _runtimeManager.getMem()},
	m_gasMeter(_gasMeter)
{}

llvm::Function* Memory::getRequireFunc()
{
	auto& func = m_require;
	if (!func)
	{
		llvm::Type* argTypes[] = {Type::RuntimePtr, Type::Word, Type::Word, Type::BytePtr, Array::getType()->getPointerTo()};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::PrivateLinkage, "mem.require", getModule());
		auto rt = func->arg_begin();
		rt->setName("rt");
		auto offset = rt->getNextNode();
		offset->setName("offset");
		auto size = offset->getNextNode();
		size->setName("size");
		auto jmpBuf = size->getNextNode();
		jmpBuf->setName("jmpBuf");
		auto mem = jmpBuf->getNextNode();
		mem->setName("mem");

		auto preBB = llvm::BasicBlock::Create(func->getContext(), "Pre", func);
		auto checkBB = llvm::BasicBlock::Create(func->getContext(), "Check", func);
		auto resizeBB = llvm::BasicBlock::Create(func->getContext(), "Resize", func);
		auto returnBB = llvm::BasicBlock::Create(func->getContext(), "Return", func);

		InsertPointGuard guard(m_builder); // Restores insert point at function exit

		// BB "Pre": Ignore checks with size 0
		m_builder.SetInsertPoint(preBB);
		auto sizeIsZero = m_builder.CreateICmpEQ(size, Constant::get(0));
		m_builder.CreateCondBr(sizeIsZero, returnBB, checkBB);

		// BB "Check"
		m_builder.SetInsertPoint(checkBB);
		auto uaddWO = llvm::Intrinsic::getDeclaration(getModule(), llvm::Intrinsic::uadd_with_overflow, Type::Word);
		auto uaddRes = m_builder.CreateCall2(uaddWO, offset, size, "res");
		auto sizeRequired = m_builder.CreateExtractValue(uaddRes, 0, "sizeReq");
		auto overflow1 = m_builder.CreateExtractValue(uaddRes, 1, "overflow1");
		auto currSize = m_memory.size(mem);
		auto tooSmall = m_builder.CreateICmpULE(m_builder.CreateZExt(currSize, Type::Word), sizeRequired, "tooSmall");
		auto resizeNeeded = m_builder.CreateOr(tooSmall, overflow1, "resizeNeeded");
		m_builder.CreateCondBr(resizeNeeded, resizeBB, returnBB); // OPT branch weights?

		// BB "Resize"
		m_builder.SetInsertPoint(resizeBB);
		// Check gas first
		uaddRes = m_builder.CreateCall2(uaddWO, sizeRequired, Constant::get(31), "res");
		auto wordsRequired = m_builder.CreateExtractValue(uaddRes, 0);
		auto overflow2 = m_builder.CreateExtractValue(uaddRes, 1, "overflow2");
		auto overflow = m_builder.CreateOr(overflow1, overflow2, "overflow");
		wordsRequired = m_builder.CreateSelect(overflow, Constant::get(-1), wordsRequired);
		wordsRequired = m_builder.CreateUDiv(wordsRequired, Constant::get(32), "wordsReq");
		sizeRequired = m_builder.CreateMul(wordsRequired, Constant::get(32), "roundedSizeReq");
		auto words = m_builder.CreateUDiv(currSize, m_builder.getInt64(32), "words");	// size is always 32*k
		auto newWords = m_builder.CreateSub(wordsRequired, m_builder.CreateZExt(words, Type::Word), "addtionalWords");
		m_gasMeter.countMemory(newWords, jmpBuf);
		// Resize
		auto extendSize = m_builder.CreateTrunc(sizeRequired, Type::Size, "extendSize");
		m_memory.extend(mem, extendSize);
		m_builder.CreateBr(returnBB);

		// BB "Return"
		m_builder.SetInsertPoint(returnBB);
		m_builder.CreateRetVoid();
	}
	return func;
}

llvm::Function* Memory::createFunc(bool _isStore, llvm::Type* _valueType)
{
	auto isWord = _valueType == Type::Word;

	llvm::Type* storeArgs[] = {Type::RuntimePtr, Type::Word, _valueType, Array::getType()->getPointerTo()};
	llvm::Type* loadArgs[] = {Type::RuntimePtr, Type::Word, Array::getType()->getPointerTo()};
	auto name = _isStore ? isWord ? "mstore" : "mstore8" : "mload";
	auto funcType = _isStore ? llvm::FunctionType::get(Type::Void, storeArgs, false) : llvm::FunctionType::get(Type::Word, loadArgs, false);
	auto func = llvm::Function::Create(funcType, llvm::Function::PrivateLinkage, name, getModule());

	InsertPointGuard guard(m_builder); // Restores insert point at function exit

	m_builder.SetInsertPoint(llvm::BasicBlock::Create(func->getContext(), {}, func));
	auto rt = func->arg_begin();
	rt->setName("rt");
	auto index = rt->getNextNode();
	index->setName("index");

	if (_isStore)
	{
		auto valueArg = index->getNextNode();
		valueArg->setName("value");
		auto mem = valueArg->getNextNode();
		mem->setName("mem");
		auto value = isWord ? Endianness::toBE(m_builder, valueArg) : valueArg;
		auto memPtr = m_memory.getPtr(mem, m_builder.CreateTrunc(index, Type::Size));
		auto valuePtr = m_builder.CreateBitCast(memPtr, _valueType->getPointerTo(), "valuePtr");
		m_builder.CreateStore(value, valuePtr);
		m_builder.CreateRetVoid();
	}
	else
	{
		auto mem = index->getNextNode();
		mem->setName("mem");
		auto memPtr = m_memory.getPtr(mem, m_builder.CreateTrunc(index, Type::Size));
		llvm::Value* ret = m_builder.CreateLoad(memPtr);
		ret = Endianness::toNative(m_builder, ret);
		m_builder.CreateRet(ret);
	}

	return func;
}

llvm::Function* Memory::getLoadWordFunc()
{
	auto& func = m_loadWord;
	if (!func)
		func = createFunc(false, Type::Word);
	return func;
}

llvm::Function* Memory::getStoreWordFunc()
{
	auto& func = m_storeWord;
	if (!func)
		func = createFunc(true, Type::Word);
	return func;
}

llvm::Function* Memory::getStoreByteFunc()
{
	auto& func = m_storeByte;
	if (!func)
		func = createFunc(true, Type::Byte);
	return func;
}


llvm::Value* Memory::loadWord(llvm::Value* _addr)
{
	require(_addr, Constant::get(Type::Word->getPrimitiveSizeInBits() / 8));
	return createCall(getLoadWordFunc(), {getRuntimeManager().getRuntimePtr(), _addr, getRuntimeManager().getMem()});
}

void Memory::storeWord(llvm::Value* _addr, llvm::Value* _word)
{
	require(_addr, Constant::get(Type::Word->getPrimitiveSizeInBits() / 8));
	createCall(getStoreWordFunc(), {getRuntimeManager().getRuntimePtr(), _addr, _word, getRuntimeManager().getMem()});
}

void Memory::storeByte(llvm::Value* _addr, llvm::Value* _word)
{
	require(_addr, Constant::get(Type::Byte->getPrimitiveSizeInBits() / 8));
	auto byte = m_builder.CreateTrunc(_word, Type::Byte, "byte");
	createCall(getStoreByteFunc(), {getRuntimeManager().getRuntimePtr(), _addr, byte, getRuntimeManager().getMem()});
}

llvm::Value* Memory::getData()
{
	auto memPtr = m_builder.CreateBitCast(getRuntimeManager().getMem(), Type::BytePtr->getPointerTo());
	auto data = m_builder.CreateLoad(memPtr, "data");
	assert(data->getType() == Type::BytePtr);
	return data;
}

llvm::Value* Memory::getSize()
{
	return m_builder.CreateZExt(m_memory.size(), Type::Word, "msize"); // TODO: Allow placing i64 on stack
}

llvm::Value* Memory::getBytePtr(llvm::Value* _index)
{
	auto idx = m_builder.CreateTrunc(_index, Type::Size, "idx"); // Never allow memory index be a type bigger than i64
	return m_builder.CreateGEP(getData(), idx, "ptr");
}

void Memory::require(llvm::Value* _offset, llvm::Value* _size)
{
	if (auto constant = llvm::dyn_cast<llvm::ConstantInt>(_size))
	{
		if (!constant->getValue())
			return;
	}
	createCall(getRequireFunc(), {getRuntimeManager().getRuntimePtr(), _offset, _size, getRuntimeManager().getJmpBuf(), getRuntimeManager().getMem()});
}

void Memory::copyBytes(llvm::Value* _srcPtr, llvm::Value* _srcSize, llvm::Value* _srcIdx,
					   llvm::Value* _destMemIdx, llvm::Value* _reqBytes)
{
	require(_destMemIdx, _reqBytes);

	// Additional copy cost
	// TODO: This round ups to 32 happens in many places
	auto reqBytes = m_builder.CreateTrunc(_reqBytes, Type::Gas);
	auto copyWords = m_builder.CreateUDiv(m_builder.CreateNUWAdd(reqBytes, m_builder.getInt64(31)), m_builder.getInt64(32));
	m_gasMeter.countCopy(copyWords);

	// Algorithm:
	// isOutsideData = idx256 >= size256
	// idx64  = trunc idx256
	// size64 = trunc size256
	// dataLeftSize = size64 - idx64  // safe if not isOutsideData
	// reqBytes64 = trunc _reqBytes   // require() handles large values
	// bytesToCopy0 = select(reqBytes64 > dataLeftSize, dataSizeLeft, reqBytes64)  // min
	// bytesToCopy = select(isOutsideData, 0, bytesToCopy0)

	auto isOutsideData = m_builder.CreateICmpUGE(_srcIdx, _srcSize);
	auto idx64 = m_builder.CreateTrunc(_srcIdx, Type::Size);
	auto size64 = m_builder.CreateTrunc(_srcSize, Type::Size);
	auto dataLeftSize = m_builder.CreateNUWSub(size64, idx64);
	auto outOfBound = m_builder.CreateICmpUGT(reqBytes, dataLeftSize);
	auto bytesToCopyInner = m_builder.CreateSelect(outOfBound, dataLeftSize, reqBytes);
	auto bytesToCopy = m_builder.CreateSelect(isOutsideData, m_builder.getInt64(0), bytesToCopyInner);

	auto src = m_builder.CreateGEP(_srcPtr, idx64, "src");
	auto dstIdx = m_builder.CreateTrunc(_destMemIdx, Type::Size, "dstIdx"); // Never allow memory index be a type bigger than i64
	auto dst = m_memory.getPtr(getRuntimeManager().getMem(), dstIdx);
	auto dst2 = m_builder.CreateGEP(getData(), dstIdx, "dst2");
	m_builder.CreateMemCpy(dst, src, bytesToCopy, 0);
	m_builder.CreateMemCpy(dst2, src, bytesToCopy, 0);
}

}
}
}
