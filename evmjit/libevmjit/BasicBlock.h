#pragma once

#include <vector>

#include "Common.h"
#include "Stack.h"

namespace dev
{
namespace eth
{
namespace jit
{
using namespace evmjit;
using instr_idx = uint64_t;

class LocalStack
{
public:
	explicit LocalStack(Stack& _globalStack);
	LocalStack(LocalStack const&) = delete;
	void operator=(LocalStack const&) = delete;

	/// Pushes value on stack
	void push(llvm::Value* _value);

	/// Pops and returns top value
	llvm::Value* pop();

	/// Duplicates _index'th value on stack
	void dup(size_t _index);

	/// Swaps _index'th value on stack with a value on stack top.
	/// @param _index Index of value to be swaped. Must be > 0.
	void swap(size_t _index);

	ssize_t size() const { return static_cast<ssize_t>(m_local.size()) - static_cast<ssize_t>(m_globalPops); }
	ssize_t minSize() const { return m_minSize; }
	ssize_t maxSize() const { return m_maxSize; }

	/// Finalize local stack: check the requirements and update of the global stack.
	void finalize(llvm::IRBuilder<>& _builder, llvm::BasicBlock& _bb);

private:
	/// Gets _index'th value from top (counting from 0)
	llvm::Value* get(size_t _index);

	/// Sets _index'th value from top (counting from 0)
	void set(size_t _index, llvm::Value* _value);

	/// Items fetched from global stack. First element matches the top of the global stack.
	/// Can contain nulls if some items has been skipped.
	std::vector<llvm::Value*> m_input;

	/// Local stack items that has not been pushed to global stack. First item is just above global stack.
	std::vector<llvm::Value*> m_local;

	Stack& m_global;			///< Reference to global stack.

	size_t m_globalPops = 0; 	///< Number of items poped from global stack. In other words: global - local stack overlap.
	ssize_t m_minSize = 0;		///< Minimum reached local stack size. Can be negative.
	ssize_t m_maxSize = 0;		///< Maximum reached local stack size.
};

class BasicBlock
{
public:
	explicit BasicBlock(instr_idx _firstInstrIdx, code_iterator _begin, code_iterator _end, llvm::Function* _mainFunc);

	llvm::BasicBlock* llvm() { return m_llvmBB; }

	instr_idx firstInstrIdx() const { return m_firstInstrIdx; }
	code_iterator begin() const { return m_begin; }
	code_iterator end() const { return m_end; }

private:
	instr_idx const m_firstInstrIdx = 0; 	///< Code index of first instruction in the block
	code_iterator const m_begin = {};		///< Iterator pointing code beginning of the block
	code_iterator const m_end = {};			///< Iterator pointing code end of the block

	llvm::BasicBlock* const m_llvmBB;		///< Reference to the LLVM BasicBlock
};

}
}
}
