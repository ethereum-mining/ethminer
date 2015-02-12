#pragma once

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/IRBuilder.h>
#include "preprocessor/llvm_includes_end.h"

namespace dev
{
namespace eth
{
namespace jit
{

struct Endianness
{
	static llvm::Value* toBE(llvm::IRBuilder<>& _builder, llvm::Value* _word) { return bswapIfLE(_builder, _word); }
	static llvm::Value* toNative(llvm::IRBuilder<>& _builder, llvm::Value* _word) { return bswapIfLE(_builder, _word); }

private:
	static llvm::Value* bswapIfLE(llvm::IRBuilder<>& _builder, llvm::Value* _word);
};

}
}
}
