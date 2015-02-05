
#include "RuntimeManager.h"

#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IntrinsicInst.h>

#include "RuntimeData.h"
#include "Instruction.h"

namespace dev
{
namespace eth
{
namespace jit
{

llvm::StructType* RuntimeManager::getRuntimeDataType()
{
	static llvm::StructType* type = nullptr;
	if (!type)
	{
		llvm::Type* elems[] =
		{
			Type::Size,		// gas
			Type::Size,		// gasPrice
			Type::BytePtr,	// callData
			Type::Size,		// callDataSize
			Type::Word,		// address
			Type::Word,		// caller
			Type::Word,		// origin
			Type::Word,		// callValue
			Type::Word,		// coinBase
			Type::Word,		// difficulty
			Type::Word,		// gasLimit
			Type::Size,		// blockNumber
			Type::Size,		// blockTimestamp
			Type::BytePtr,	// code
			Type::Size,		// codeSize
		};
		type = llvm::StructType::create(elems, "RuntimeData");
	}
	return type;
}

llvm::StructType* RuntimeManager::getRuntimeType()
{
	static llvm::StructType* type = nullptr;
	if (!type)
	{
		llvm::Type* elems[] =
		{
			Type::RuntimeDataPtr,	// data
			Type::EnvPtr,			// Env*
			Type::BytePtr,			// jmpbuf
			Type::BytePtr,			// memory data
			Type::Word,				// memory size
		};
		type = llvm::StructType::create(elems, "Runtime");
	}
	return type;
}

namespace
{
llvm::Twine getName(RuntimeData::Index _index)
{
	switch (_index)
	{
	default:						return "data";
	case RuntimeData::Address:		return "address";
	case RuntimeData::Caller:		return "caller";
	case RuntimeData::Origin:		return "origin";
	case RuntimeData::CallValue:	return "callvalue";
	case RuntimeData::GasPrice:		return "gasprice";
	case RuntimeData::CoinBase:		return "coinbase";
	case RuntimeData::Difficulty:	return "difficulty";
	case RuntimeData::GasLimit:		return "gaslimit";
	case RuntimeData::CallData:		return "callData";
	case RuntimeData::Code:			return "code";
	case RuntimeData::CodeSize:		return "code";
	case RuntimeData::CallDataSize:	return "callDataSize";
	case RuntimeData::Gas:			return "gas";
	case RuntimeData::Number:	return "number";
	case RuntimeData::Timestamp:	return "timestamp";
	}
}
}

RuntimeManager::RuntimeManager(llvm::IRBuilder<>& _builder): CompilerHelper(_builder)
{
	m_longjmp = llvm::Intrinsic::getDeclaration(getModule(), llvm::Intrinsic::longjmp);

	// Unpack data
	auto rtPtr = getRuntimePtr();
	m_dataPtr = m_builder.CreateLoad(m_builder.CreateStructGEP(rtPtr, 0), "data");
	assert(m_dataPtr->getType() == Type::RuntimeDataPtr);
	m_envPtr = m_builder.CreateLoad(m_builder.CreateStructGEP(rtPtr, 1), "env");
	assert(m_envPtr->getType() == Type::EnvPtr);
}

llvm::Value* RuntimeManager::getRuntimePtr()
{
	// Expect first argument of a function to be a pointer to Runtime
	auto func = m_builder.GetInsertBlock()->getParent();
	auto rtPtr = &func->getArgumentList().front();
	assert(rtPtr->getType() == Type::RuntimePtr);
	return rtPtr;
}

llvm::Value* RuntimeManager::getDataPtr()
{
	if (getMainFunction())
		return m_dataPtr;

	auto rtPtr = getRuntimePtr();
	auto dataPtr = m_builder.CreateLoad(m_builder.CreateStructGEP(rtPtr, 0), "data");
	assert(dataPtr->getType() == getRuntimeDataType()->getPointerTo());
	return dataPtr;
}

llvm::Value* RuntimeManager::getEnvPtr()
{
	assert(getMainFunction());	// Available only in main function
	return m_envPtr;
}

llvm::Value* RuntimeManager::getPtr(RuntimeData::Index _index)
{
	auto ptr = getBuilder().CreateStructGEP(getDataPtr(), _index);
	assert(getRuntimeDataType()->getElementType(_index)->getPointerTo() == ptr->getType());
	return ptr;
}

llvm::Value* RuntimeManager::get(RuntimeData::Index _index)
{
	return getBuilder().CreateLoad(getPtr(_index), getName(_index));
}

void RuntimeManager::set(RuntimeData::Index _index, llvm::Value* _value)
{
	auto ptr = getPtr(_index);
	assert(ptr->getType() == _value->getType()->getPointerTo());
	getBuilder().CreateStore(_value, ptr);
}

void RuntimeManager::registerReturnData(llvm::Value* _offset, llvm::Value* _size)
{
	auto memPtr = getBuilder().CreateStructGEP(getRuntimePtr(), 3);
	auto mem = getBuilder().CreateLoad(memPtr, "memory");
	auto idx = m_builder.CreateTrunc(_offset, Type::Size, "idx"); // Never allow memory index be a type bigger than i64 // TODO: Report bug & fix to LLVM
	auto returnDataPtr = getBuilder().CreateGEP(mem, idx);
	set(RuntimeData::ReturnData, returnDataPtr);

	auto size64 = getBuilder().CreateTrunc(_size, Type::Size);
	set(RuntimeData::ReturnDataSize, size64);
}

void RuntimeManager::registerSuicide(llvm::Value* _balanceAddress)
{
	set(RuntimeData::SuicideDestAddress, _balanceAddress);
}

void RuntimeManager::raiseException(ReturnCode _returnCode)
{
	m_builder.CreateCall2(m_longjmp, getJmpBuf(), Constant::get(_returnCode));
}

llvm::Value* RuntimeManager::get(Instruction _inst)
{
	switch (_inst)
	{
	default: assert(false); return nullptr;
	case Instruction::ADDRESS:		return get(RuntimeData::Address);
	case Instruction::CALLER:		return get(RuntimeData::Caller);
	case Instruction::ORIGIN:		return get(RuntimeData::Origin);
	case Instruction::CALLVALUE:	return get(RuntimeData::CallValue);
	case Instruction::GASPRICE:		return get(RuntimeData::GasPrice);
	case Instruction::COINBASE:		return get(RuntimeData::CoinBase);
	case Instruction::DIFFICULTY:	return get(RuntimeData::Difficulty);
	case Instruction::GASLIMIT:		return get(RuntimeData::GasLimit);
	case Instruction::NUMBER:		return get(RuntimeData::Number);
	case Instruction::TIMESTAMP:	return get(RuntimeData::Timestamp);
	}
}

llvm::Value* RuntimeManager::getCallData()
{
	return get(RuntimeData::CallData);
}

llvm::Value* RuntimeManager::getCode()
{
	return get(RuntimeData::Code);
}

llvm::Value* RuntimeManager::getCodeSize()
{
	auto value = get(RuntimeData::CodeSize);
	assert(value->getType() == Type::Size);
	return getBuilder().CreateZExt(value, Type::Word);
}

llvm::Value* RuntimeManager::getCallDataSize()
{
	auto value = get(RuntimeData::CallDataSize);
	assert(value->getType() == Type::Size);
	return getBuilder().CreateZExt(value, Type::Word);
}

llvm::Value* RuntimeManager::getJmpBuf()
{
	auto ptr = getBuilder().CreateStructGEP(getRuntimePtr(), 2, "jmpbufPtr");
	return getBuilder().CreateLoad(ptr, "jmpbuf");
}

llvm::Value* RuntimeManager::getGas()
{
	auto value = get(RuntimeData::Gas);
	assert(value->getType() == Type::Size);
	return getBuilder().CreateZExt(value, Type::Word);
}

void RuntimeManager::setGas(llvm::Value* _gas)
{
	auto newGas = getBuilder().CreateTrunc(_gas, Type::Size);
	set(RuntimeData::Gas, newGas);
}

}
}
}
