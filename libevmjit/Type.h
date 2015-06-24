#pragma once

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Type.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Metadata.h>
#include "preprocessor/llvm_includes_end.h"

#include "evmjit/JIT.h" // ReturnCode

namespace dev
{
namespace eth
{
namespace jit
{
using namespace evmjit;

struct Type
{
	static llvm::IntegerType* Word;
	static llvm::PointerType* WordPtr;

	static llvm::IntegerType* Bool;
	static llvm::IntegerType* Size;
	static llvm::IntegerType* Gas;
	static llvm::PointerType* GasPtr;

	static llvm::IntegerType* Byte;
	static llvm::PointerType* BytePtr;

	static llvm::Type* Void;

	/// Main function return type
	static llvm::IntegerType* MainReturn;

	static llvm::PointerType* EnvPtr;
	static llvm::PointerType* RuntimeDataPtr;
	static llvm::PointerType* RuntimePtr;

	// TODO: Redesign static LLVM objects
	static llvm::MDNode* expectTrue;

	static void init(llvm::LLVMContext& _context);
};

struct Constant
{
	static llvm::ConstantInt* gasMax;

	/// Returns word-size constant
	static llvm::ConstantInt* get(int64_t _n);
	static llvm::ConstantInt* get(llvm::APInt const& _n);

	static llvm::ConstantInt* get(ReturnCode _returnCode);
};

}
}
}

