#include "Memory.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/IntrinsicInst.h>
#include "preprocessor/llvm_includes_end.h"

#include "Type.h"
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
		llvm::Type* argTypes[] = {Array::getType()->getPointerTo(), Type::Word, Type::Word, Type::BytePtr, Type::GasPtr};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::PrivateLinkage, "mem.require", getModule());
		func->setDoesNotThrow();

		auto mem = &func->getArgumentList().front();
		mem->setName("mem");
		auto blkOffset = mem->getNextNode();
		blkOffset->setName("blkOffset");
		auto blkSize = blkOffset->getNextNode();
		blkSize->setName("blkSize");
		auto jmpBuf = blkSize->getNextNode();
		jmpBuf->setName("jmpBuf");
		auto gas = jmpBuf->getNextNode();
		gas->setName("gas");

		auto preBB = llvm::BasicBlock::Create(func->getContext(), "Pre", func);
		auto checkBB = llvm::BasicBlock::Create(func->getContext(), "Check", func);
		auto resizeBB = llvm::BasicBlock::Create(func->getContext(), "Resize", func);
		auto returnBB = llvm::BasicBlock::Create(func->getContext(), "Return", func);

		InsertPointGuard guard(m_builder); // Restores insert point at function exit

		// BB "Pre": Ignore checks with size 0
		m_builder.SetInsertPoint(preBB);
		m_builder.CreateCondBr(m_builder.CreateICmpNE(blkSize, Constant::get(0)), checkBB, returnBB, Type::expectTrue);

		// BB "Check"
		m_builder.SetInsertPoint(checkBB);
		static const auto c_inputMax = uint64_t(1) << 33; // max value of blkSize and blkOffset that will not result in integer overflow in calculations below
		auto blkOffsetOk = m_builder.CreateICmpULE(blkOffset, Constant::get(c_inputMax), "blkOffsetOk");
		auto blkO = m_builder.CreateSelect(blkOffsetOk, m_builder.CreateTrunc(blkOffset, Type::Size), m_builder.getInt64(c_inputMax), "bklO");
		auto blkSizeOk = m_builder.CreateICmpULE(blkSize, Constant::get(c_inputMax), "blkSizeOk");
		auto blkS = m_builder.CreateSelect(blkSizeOk, m_builder.CreateTrunc(blkSize, Type::Size), m_builder.getInt64(c_inputMax), "bklS");

		auto sizeReq0 = m_builder.CreateNUWAdd(blkO, blkS, "sizeReq0");
		auto sizeReq = m_builder.CreateAnd(m_builder.CreateNUWAdd(sizeReq0, m_builder.getInt64(31)), uint64_t(-1) << 5, "sizeReq"); // s' = ((s0 + 31) / 32) * 32
		auto sizeCur = m_memory.size(mem);
		auto sizeOk = m_builder.CreateICmpULE(sizeReq, sizeCur, "sizeOk");

		m_builder.CreateCondBr(sizeOk, returnBB, resizeBB, Type::expectTrue);

		// BB "Resize"
		m_builder.SetInsertPoint(resizeBB);
		// Check gas first
		auto w1 = m_builder.CreateLShr(sizeReq, 5);
		auto w1s = m_builder.CreateNUWMul(w1, w1);
		auto c1 = m_builder.CreateAdd(m_builder.CreateNUWMul(w1, m_builder.getInt64(3)), m_builder.CreateLShr(w1s, 9));
		auto w0 = m_builder.CreateLShr(sizeCur, 5);
		auto w0s = m_builder.CreateNUWMul(w0, w0);
		auto c0 = m_builder.CreateAdd(m_builder.CreateNUWMul(w0, m_builder.getInt64(3)), m_builder.CreateLShr(w0s, 9));
		auto cc = m_builder.CreateNUWSub(c1, c0);
		auto costOk = m_builder.CreateAnd(blkOffsetOk, blkSizeOk, "costOk");
		auto c = m_builder.CreateSelect(costOk, cc, m_builder.getInt64(std::numeric_limits<int64_t>::max()), "c");
		m_gasMeter.count(c, jmpBuf, gas);
		// Resize
		m_memory.extend(mem, sizeReq);
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

	llvm::Type* storeArgs[] = {Array::getType()->getPointerTo(), Type::Word, _valueType};
	llvm::Type* loadArgs[] = {Array::getType()->getPointerTo(), Type::Word};
	auto name = _isStore ? isWord ? "mstore" : "mstore8" : "mload";
	auto funcType = _isStore ? llvm::FunctionType::get(Type::Void, storeArgs, false) : llvm::FunctionType::get(Type::Word, loadArgs, false);
	auto func = llvm::Function::Create(funcType, llvm::Function::PrivateLinkage, name, getModule());

	InsertPointGuard guard(m_builder); // Restores insert point at function exit

	m_builder.SetInsertPoint(llvm::BasicBlock::Create(func->getContext(), {}, func));
	auto mem = &func->getArgumentList().front();
	mem->setName("mem");
	auto index = mem->getNextNode();
	index->setName("index");

	if (_isStore)
	{
		auto valueArg = index->getNextNode();
		valueArg->setName("value");
		auto value = isWord ? Endianness::toBE(m_builder, valueArg) : valueArg;
		auto memPtr = m_memory.getPtr(mem, m_builder.CreateTrunc(index, Type::Size));
		auto valuePtr = m_builder.CreateBitCast(memPtr, _valueType->getPointerTo(), "valuePtr");
		m_builder.CreateStore(value, valuePtr);
		m_builder.CreateRetVoid();
	}
	else
	{
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
	return createCall(getLoadWordFunc(), {getRuntimeManager().getMem(), _addr});
}

void Memory::storeWord(llvm::Value* _addr, llvm::Value* _word)
{
	require(_addr, Constant::get(Type::Word->getPrimitiveSizeInBits() / 8));
	createCall(getStoreWordFunc(), {getRuntimeManager().getMem(), _addr, _word});
}

void Memory::storeByte(llvm::Value* _addr, llvm::Value* _word)
{
	require(_addr, Constant::get(Type::Byte->getPrimitiveSizeInBits() / 8));
	auto byte = m_builder.CreateTrunc(_word, Type::Byte, "byte");
	createCall(getStoreByteFunc(), {getRuntimeManager().getMem(), _addr, byte});
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
	return m_builder.CreateGEP(getData(), _index, "ptr");
}

void Memory::require(llvm::Value* _offset, llvm::Value* _size)
{
	if (auto constant = llvm::dyn_cast<llvm::ConstantInt>(_size))
	{
		if (!constant->getValue())
			return;
	}
	createCall(getRequireFunc(), {getRuntimeManager().getMem(), _offset, _size, getRuntimeManager().getJmpBuf(), getRuntimeManager().getGasPtr()});
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
	auto bytesToCopy = m_builder.CreateSelect(isOutsideData, m_builder.getInt64(0), bytesToCopyInner, "bytesToCopy");
	auto bytesToZero = m_builder.CreateNUWSub(reqBytes, bytesToCopy, "bytesToZero");

	auto src = m_builder.CreateGEP(_srcPtr, idx64, "src");
	auto dstIdx = m_builder.CreateTrunc(_destMemIdx, Type::Size, "dstIdx");
	auto padIdx = m_builder.CreateNUWAdd(dstIdx, bytesToCopy, "padIdx");
	auto dst = m_memory.getPtr(getRuntimeManager().getMem(), dstIdx);
	auto pad = m_memory.getPtr(getRuntimeManager().getMem(), padIdx);
	m_builder.CreateMemCpy(dst, src, bytesToCopy, 0);
	m_builder.CreateMemSet(pad, m_builder.getInt8(0), bytesToZero, 0);
}

}
}
}
