#include "Optimizer.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include "preprocessor/llvm_includes_end.h"

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

}
}
}
