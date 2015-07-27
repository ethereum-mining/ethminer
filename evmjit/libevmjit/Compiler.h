#pragma once

#include "BasicBlock.h"

namespace dev
{
namespace eth
{
namespace jit
{

class Compiler
{
public:

	struct Options
	{
		/// Rewrite switch instructions to sequences of branches
		bool rewriteSwitchToBranches = true;

		/// Dump CFG as a .dot file for graphviz
		bool dumpCFG = false;
	};

	Compiler(Options const& _options);

	std::unique_ptr<llvm::Module> compile(code_iterator _begin, code_iterator _end, std::string const& _id);

private:

	std::vector<BasicBlock> createBasicBlocks(code_iterator _begin, code_iterator _end, llvm::SwitchInst& _switchInst);

	void compileBasicBlock(BasicBlock& _basicBlock, class RuntimeManager& _runtimeManager, class Arith256& _arith, class Memory& _memory, class Ext& _ext, class GasMeter& _gasMeter,
			llvm::BasicBlock* _nextBasicBlock, class Stack& _globalStack, llvm::SwitchInst& _jumpTable);

	void fillJumpTable();

	/// Compiler options
	Options const& m_options;

	/// Helper class for generating IR
	llvm::IRBuilder<> m_builder;

	/// Stop basic block - terminates execution with STOP code (0)
	llvm::BasicBlock* m_stopBB = nullptr;

	/// Abort basic block - terminates execution with OOG-like state
	llvm::BasicBlock* m_abortBB = nullptr;

	/// Block with a jump table.
	llvm::BasicBlock* m_jumpTableBB = nullptr;

	/// Main program function
	llvm::Function* m_mainFunc = nullptr; // TODO: Remove
};

}
}
}
