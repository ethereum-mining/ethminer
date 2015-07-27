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
	pm.add(llvm::createFunctionInliningPass(2, 2));
	pm.add(llvm::createCFGSimplificationPass());
	pm.add(llvm::createInstructionCombiningPass());
	pm.add(llvm::createAggressiveDCEPass());
	pm.add(llvm::createLowerSwitchPass());
	return pm.run(_module);
}

namespace
{

class LowerEVMPass: public llvm::BasicBlockPass
{
	static char ID;

public:
	LowerEVMPass():
		llvm::BasicBlockPass(ID)
	{}

	virtual bool runOnBasicBlock(llvm::BasicBlock& _bb) override;

	using llvm::BasicBlockPass::doFinalization;
	virtual bool doFinalization(llvm::Module& _module) override;
};

char LowerEVMPass::ID = 0;

bool LowerEVMPass::runOnBasicBlock(llvm::BasicBlock& _bb)
{
	auto modified = false;
	auto module = _bb.getParent()->getParent();
	auto i512Ty = llvm::IntegerType::get(_bb.getContext(), 512);
	for (auto it = _bb.begin(); it != _bb.end(); )
	{
		auto& inst = *it++;
		llvm::Function* func = nullptr;
		if (inst.getType() == Type::Word)
		{
			switch (inst.getOpcode())
			{
			case llvm::Instruction::Mul:
				func = Arith256::getMulFunc(*module);
				break;

			case llvm::Instruction::UDiv:
				func = Arith256::getUDiv256Func(*module);
				break;

			case llvm::Instruction::URem:
				func = Arith256::getURem256Func(*module);
				break;

			case llvm::Instruction::SDiv:
				func = Arith256::getSDiv256Func(*module);
				break;

			case llvm::Instruction::SRem:
				func = Arith256::getSRem256Func(*module);
				break;
			}
		}
		else if (inst.getType() == i512Ty)
		{
			switch (inst.getOpcode())
			{
			case llvm::Instruction::URem:
				func = Arith256::getURem512Func(*module);
				break;
			}
		}

		if (func)
		{
			auto call = llvm::CallInst::Create(func, {inst.getOperand(0), inst.getOperand(1)}, "", &inst);
			inst.replaceAllUsesWith(call);
			inst.eraseFromParent();
			modified = true;
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
	pm.add(llvm::createCFGSimplificationPass());
	pm.add(llvm::createDeadCodeEliminationPass());
	pm.add(new LowerEVMPass{});
	return pm.run(_module);
}

}
}
}
