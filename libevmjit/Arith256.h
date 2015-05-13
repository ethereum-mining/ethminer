#pragma once

#include "CompilerHelper.h"

namespace dev
{
namespace eth
{
namespace jit
{

class Arith256 : public CompilerHelper
{
public:
	Arith256(llvm::IRBuilder<>& _builder);

	std::pair<llvm::Value*, llvm::Value*> div(llvm::Value* _arg1, llvm::Value* _arg2);
	std::pair<llvm::Value*, llvm::Value*> sdiv(llvm::Value* _arg1, llvm::Value* _arg2);
	llvm::Value* exp(llvm::Value* _arg1, llvm::Value* _arg2);
	llvm::Value* mulmod(llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3);
	llvm::Value* addmod(llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3);

	void debug(llvm::Value* _value, char _c);

	static llvm::Function* getMulFunc(llvm::Module& _module);
	static llvm::Function* getUDiv256Func(llvm::Module& _module);
	static llvm::Function* getUDivRem256Func(llvm::Module& _module);

private:
	llvm::Function* getMul512Func();
	llvm::Function* getDivFunc(llvm::Type* _type);
	llvm::Function* getExpFunc();
	llvm::Function* getAddModFunc();
	llvm::Function* getMulModFunc();

	llvm::Function* m_mul512 = nullptr;
	llvm::Function* m_div = nullptr;
	llvm::Function* m_div512 = nullptr;
	llvm::Function* m_exp = nullptr;
	llvm::Function* m_addmod = nullptr;
	llvm::Function* m_mulmod = nullptr;
	llvm::Function* m_debug = nullptr;
};


}
}
}
