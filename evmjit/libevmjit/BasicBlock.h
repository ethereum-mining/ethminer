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

class BasicBlock
{
public:
	class LocalStack
	{
	public:
		/// Pushes value on stack
		void push(llvm::Value* _value);

		/// Pops and returns top value
		llvm::Value* pop();

		/// Duplicates _index'th value on stack
		void dup(size_t _index);

		/// Swaps _index'th value on stack with a value on stack top.
		/// @param _index Index of value to be swaped. Must be > 0.
		void swap(size_t _index);

		size_t getMaxSize() const { return m_maxSize; }
		int getDiff() const { return m_bblock.m_tosOffset; }

	private:
		LocalStack(BasicBlock& _owner);
		LocalStack(LocalStack const&) = delete;
		void operator=(LocalStack const&) = delete;
		friend BasicBlock;

		/// Gets _index'th value from top (counting from 0)
		llvm::Value* get(size_t _index);

		/// Sets _index'th value from top (counting from 0)
		void set(size_t _index, llvm::Value* _value);

		std::vector<llvm::Value*>::iterator getItemIterator(size_t _index);

	private:
		BasicBlock& m_bblock;
		int m_maxSize = 0; ///< Max size reached by the stack.
	};

	explicit BasicBlock(instr_idx _firstInstrIdx, code_iterator _begin, code_iterator _end, llvm::Function* _mainFunc, llvm::IRBuilder<>& _builder, bool isJumpDest);
	explicit BasicBlock(std::string _name, llvm::Function* _mainFunc, llvm::IRBuilder<>& _builder, bool isJumpDest);

	BasicBlock(const BasicBlock&) = delete;
	BasicBlock& operator=(const BasicBlock&) = delete;

	llvm::BasicBlock* llvm() { return m_llvmBB; }

	instr_idx firstInstrIdx() const { return m_firstInstrIdx; }
	code_iterator begin() const { return m_begin; }
	code_iterator end() const { return m_end; }

	bool isJumpDest() const { return m_isJumpDest; }

	llvm::Value* getJumpTarget() const { return m_jumpTarget; }
	void setJumpTarget(llvm::Value* _jumpTarget) { m_jumpTarget = _jumpTarget; }

	LocalStack& localStack() { return m_stack; }

	/// Optimization: propagates values between local stacks in basic blocks
	/// to avoid excessive pushing/popping on the EVM stack.
	static void linkLocalStacks(std::vector<BasicBlock*> _basicBlocks, llvm::IRBuilder<>& _builder);

	/// Synchronize current local stack with the EVM stack.
	void synchronizeLocalStack(Stack& _evmStack);

	/// Prints local stack and block instructions to stderr.
	/// Useful for calling in a debugger session.
	void dump();
	void dump(std::ostream& os, bool _dotOutput = false);

private:
	instr_idx const m_firstInstrIdx = 0; 	///< Code index of first instruction in the block
	code_iterator const m_begin = {};			///< Iterator pointing code beginning of the block
	code_iterator const m_end = {};				///< Iterator pointing code end of the block

	llvm::BasicBlock* const m_llvmBB;

	/// Basic black state vector (stack) - current/end values and their positions on stack
	/// @internal Must be AFTER m_llvmBB
	LocalStack m_stack;

	llvm::IRBuilder<>& m_builder;

	/// This stack contains LLVM values that correspond to items found at
	/// the EVM stack when the current basic block starts executing.
	/// Location 0 corresponds to the top of the EVM stack, location 1 is
	/// the item below the top and so on. The stack grows as the code
	/// accesses more items on the EVM stack but once a value is put on
	/// the stack, it will never be replaced.
	std::vector<llvm::Value*> m_initialStack;

	/// This stack tracks the contents of the EVM stack as the basic block
	/// executes. It may grow on both sides, as the code pushes items on
	/// top of the stack or changes existing items.
	std::vector<llvm::Value*> m_currentStack;

	/// How many items higher is the current stack than the initial one.
	/// May be negative.
	int m_tosOffset = 0;

	/// Is the basic block a valid jump destination.
	/// JUMPDEST is the first instruction of the basic block.
	bool const m_isJumpDest = false;

	/// If block finishes with dynamic jump target index is stored here
	llvm::Value* m_jumpTarget = nullptr;
};

}
}
}
