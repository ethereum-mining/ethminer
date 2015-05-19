#pragma once

#include <array>

#include "CompilerHelper.h"
#include "Type.h"
#include "Instruction.h"

namespace dev
{
namespace eth
{
namespace jit
{
using namespace evmjit;
class Stack;

class RuntimeManager: public CompilerHelper
{
public:
	RuntimeManager(llvm::IRBuilder<>& _builder, code_iterator _codeBegin, code_iterator _codeEnd);

	llvm::Value* getRuntimePtr();
	llvm::Value* getDataPtr();
	llvm::Value* getEnvPtr();

	llvm::Value* get(RuntimeData::Index _index);
	llvm::Value* get(Instruction _inst);
	llvm::Value* getGas();
	llvm::Value* getGasPtr();
	llvm::Value* getCallData();
	llvm::Value* getCode();
	llvm::Value* getCodeSize();
	llvm::Value* getCallDataSize();
	llvm::Value* getJmpBuf() { return m_jmpBuf; }
	void setGas(llvm::Value* _gas);

	llvm::Value* getMem();

	void registerReturnData(llvm::Value* _index, llvm::Value* _size); // TODO: Move to Memory.
	void registerSuicide(llvm::Value* _balanceAddress);

	void exit(ReturnCode _returnCode);

	void abort(llvm::Value* _jmpBuf);

	void setStack(Stack& _stack) { m_stack = &_stack; }
	void setJmpBuf(llvm::Value* _jmpBuf) { m_jmpBuf = _jmpBuf; }

	static llvm::StructType* getRuntimeType();
	static llvm::StructType* getRuntimeDataType();

	void checkStackLimit(size_t _max, int _diff);

private:
	llvm::Value* getPtr(RuntimeData::Index _index);
	void set(RuntimeData::Index _index, llvm::Value* _value);

	llvm::Function* m_longjmp = nullptr;
	llvm::Value* m_jmpBuf = nullptr;
	llvm::Value* m_dataPtr = nullptr;
	llvm::Value* m_gasPtr = nullptr;
	llvm::Value* m_memPtr = nullptr;
	llvm::Value* m_envPtr = nullptr;

	std::array<llvm::Value*, static_cast<size_t>(RuntimeData::Index::CodeSize) + 1> m_dataElts;

	llvm::Value* m_stackSize = nullptr;
	llvm::Function* m_checkStackLimit = nullptr;

	code_iterator m_codeBegin = {};
	code_iterator m_codeEnd = {};

	Stack* m_stack = nullptr;
};

}
}
}
