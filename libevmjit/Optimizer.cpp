#include "Optimizer.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include "preprocessor/llvm_includes_end.h"

#include "Arith256.h"
#include "Type.h"

namespace dev
{
namespace eth
{
namespace jit
{

bool optimize(llvm::Module& _module)
{
	auto pm = llvm::legacy::PassManager{};
	//pm.add(llvm::createFunctionInliningPass(2, 2)); // Problem with APInt value bigger than 64bit
	pm.add(llvm::createCFGSimplificationPass());
	pm.add(llvm::createInstructionCombiningPass());
	pm.add(llvm::createAggressiveDCEPass());
	pm.add(llvm::createLowerSwitchPass());
	return pm.run(_module);
}

namespace
{

class LowerEVMPass : public llvm::BasicBlockPass
{
	static char ID;

	bool m_mulFuncNeeded = false;

public:
	LowerEVMPass():
		llvm::BasicBlockPass(ID)
	{}

	virtual bool runOnBasicBlock(llvm::BasicBlock& _bb) override;

	virtual bool doFinalization(llvm::Module& _module) override;
};

char LowerEVMPass::ID = 0;

bool LowerEVMPass::runOnBasicBlock(llvm::BasicBlock& _bb)
{
	auto modified = false;
	auto module = _bb.getParent()->getParent();
	for (auto&& inst : _bb)
	{
		if (inst.getOpcode() == llvm::Instruction::Mul)
		{
			if (inst.getType() == Type::Word)
			{
				auto call = llvm::CallInst::Create(Arith256::getMulFunc(*module), {inst.getOperand(0), inst.getOperand(1)}, "", &inst);
				inst.replaceAllUsesWith(call);
				modified = true;
			}
		}
	}
	return modified;
}

bool LowerEVMPass::doFinalization(llvm::Module&)
{
	return false;
}

}

bool prepare(llvm::Module& _module)
{
	auto pm = llvm::legacy::PassManager{};
	pm.add(new LowerEVMPass{});
	return pm.run(_module);
}

}
}
}
