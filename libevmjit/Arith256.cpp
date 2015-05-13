#include "Arith256.h"

#include <iostream>
#include <iomanip>

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/Module.h>
#include <llvm/IR/IntrinsicInst.h>
#include "preprocessor/llvm_includes_end.h"

#include "Type.h"
#include "Endianness.h"
#include "Utils.h"

namespace dev
{
namespace eth
{
namespace jit
{

Arith256::Arith256(llvm::IRBuilder<>& _builder) :
	CompilerHelper(_builder)
{}

void Arith256::debug(llvm::Value* _value, char _c)
{
	if (!m_debug)
	{
		llvm::Type* argTypes[] = {Type::Word, m_builder.getInt8Ty()};
		m_debug = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::ExternalLinkage, "debug", getModule());
	}
	createCall(m_debug, {m_builder.CreateZExtOrTrunc(_value, Type::Word), m_builder.getInt8(_c)});
}

llvm::Function* Arith256::getMulFunc(llvm::Module& _module)
{
	static const auto funcName = "evm.mul.i256";
	if (auto func = _module.getFunction(funcName))
		return func;

	llvm::Type* argTypes[] = {Type::Word, Type::Word};
	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Word, argTypes, false), llvm::Function::PrivateLinkage, funcName, &_module);
	func->setDoesNotThrow();
	func->setDoesNotAccessMemory();

	auto x = &func->getArgumentList().front();
	x->setName("x");
	auto y = x->getNextNode();
	y->setName("y");

	auto bb = llvm::BasicBlock::Create(_module.getContext(), {}, func);
	auto builder = llvm::IRBuilder<>{bb};
	auto i64 = Type::Size;
	auto i128 = builder.getIntNTy(128);
	auto i256 = Type::Word;
	auto c64 = Constant::get(64);
	auto c128 = Constant::get(128);
	auto c192 = Constant::get(192);

	auto x_lo = builder.CreateTrunc(x, i64, "x.lo");
	auto y_lo = builder.CreateTrunc(y, i64, "y.lo");
	auto x_mi = builder.CreateTrunc(builder.CreateLShr(x, c64), i64);
	auto y_mi = builder.CreateTrunc(builder.CreateLShr(y, c64), i64);
	auto x_hi = builder.CreateTrunc(builder.CreateLShr(x, c128), i128);
	auto y_hi = builder.CreateTrunc(builder.CreateLShr(y, c128), i128);

	auto t1 = builder.CreateMul(builder.CreateZExt(x_lo, i128), builder.CreateZExt(y_lo, i128));
	auto t2 = builder.CreateMul(builder.CreateZExt(x_lo, i128), builder.CreateZExt(y_mi, i128));
	auto t3 = builder.CreateMul(builder.CreateZExt(x_lo, i128), y_hi);
	auto t4 = builder.CreateMul(builder.CreateZExt(x_mi, i128), builder.CreateZExt(y_lo, i128));
	auto t5 = builder.CreateMul(builder.CreateZExt(x_mi, i128), builder.CreateZExt(y_mi, i128));
	auto t6 = builder.CreateMul(builder.CreateZExt(x_mi, i128), y_hi);
	auto t7 = builder.CreateMul(x_hi, builder.CreateZExt(y_lo, i128));
	auto t8 = builder.CreateMul(x_hi, builder.CreateZExt(y_mi, i128));

	auto p = builder.CreateZExt(t1, i256);
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t2, i256), c64));
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t3, i256), c128));
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t4, i256), c64));
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t5, i256), c128));
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t6, i256), c192));
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t7, i256), c128));
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t8, i256), c192));
	builder.CreateRet(p);
	return func;
}

llvm::Function* Arith256::getMul512Func(llvm::Module& _module)
{
	static const auto funcName = "evm.mul.i512";
	if (auto func = _module.getFunction(funcName))
		return func;

	auto i512Ty = llvm::IntegerType::get(_module.getContext(), 512);
	auto func = llvm::Function::Create(llvm::FunctionType::get(i512Ty, {Type::Word, Type::Word}, false), llvm::Function::PrivateLinkage, funcName, &_module);
	func->setDoesNotThrow();
	func->setDoesNotAccessMemory();

	auto x = &func->getArgumentList().front();
	x->setName("x");
	auto y = x->getNextNode();
	y->setName("y");

	auto bb = llvm::BasicBlock::Create(_module.getContext(), {}, func);
	auto builder = llvm::IRBuilder<>{bb};

	auto i128 = builder.getIntNTy(128);
	auto i256 = Type::Word;
	auto x_lo = builder.CreateZExt(builder.CreateTrunc(x, i128, "x.lo"), i256);
	auto y_lo = builder.CreateZExt(builder.CreateTrunc(y, i128, "y.lo"), i256);
	auto x_hi = builder.CreateZExt(builder.CreateTrunc(builder.CreateLShr(x, Constant::get(128)), i128, "x.hi"), i256);
	auto y_hi = builder.CreateZExt(builder.CreateTrunc(builder.CreateLShr(y, Constant::get(128)), i128, "y.hi"), i256);

	auto mul256Func = getMulFunc(_module);
	auto t1 = builder.CreateCall(mul256Func, {x_lo, y_lo});
	auto t2 = builder.CreateCall(mul256Func, {x_lo, y_hi});
	auto t3 = builder.CreateCall(mul256Func, {x_hi, y_lo});
	auto t4 = builder.CreateCall(mul256Func, {x_hi, y_hi});

	auto p = builder.CreateZExt(t1, i512Ty);
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t2, i512Ty), builder.getIntN(512, 128)));
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t3, i512Ty), builder.getIntN(512, 128)));
	p = builder.CreateAdd(p, builder.CreateShl(builder.CreateZExt(t4, i512Ty), builder.getIntN(512, 256)));
	builder.CreateRet(p);

	return func;
}

namespace
{
llvm::Function* createUDivRemFunc(llvm::Type* _type, llvm::Module& _module, char const* _funcName)
{
	// Based of "Improved shift divisor algorithm" from "Software Integer Division" by Microsoft Research
	// The following algorithm also handles divisor of value 0 returning 0 for both quotient and reminder

	auto retType = llvm::VectorType::get(_type, 2);
	auto func = llvm::Function::Create(llvm::FunctionType::get(retType, {_type, _type}, false), llvm::Function::PrivateLinkage, _funcName, &_module);
	func->setDoesNotThrow();
	func->setDoesNotAccessMemory();

	auto zero = llvm::ConstantInt::get(_type, 0);
	auto one = llvm::ConstantInt::get(_type, 1);

	auto x = &func->getArgumentList().front();
	x->setName("x");
	auto yArg = x->getNextNode();
	yArg->setName("y");

	auto entryBB = llvm::BasicBlock::Create(_module.getContext(), "Entry", func);
	auto mainBB = llvm::BasicBlock::Create(_module.getContext(), "Main", func);
	auto loopBB = llvm::BasicBlock::Create(_module.getContext(), "Loop", func);
	auto continueBB = llvm::BasicBlock::Create(_module.getContext(), "Continue", func);
	auto returnBB = llvm::BasicBlock::Create(_module.getContext(), "Return", func);

	auto builder = llvm::IRBuilder<>{entryBB};
	auto yLEx = builder.CreateICmpULE(yArg, x);
	auto r0 = x;
	builder.CreateCondBr(yLEx, mainBB, returnBB);

	builder.SetInsertPoint(mainBB);
	auto ctlzIntr = llvm::Intrinsic::getDeclaration(&_module, llvm::Intrinsic::ctlz, _type);
	// both y and r are non-zero
	auto yLz = builder.CreateCall2(ctlzIntr, yArg, builder.getInt1(true), "y.lz");
	auto rLz = builder.CreateCall2(ctlzIntr, r0, builder.getInt1(true), "r.lz");
	auto i0 = builder.CreateNUWSub(yLz, rLz, "i0");
	auto y0 = builder.CreateShl(yArg, i0);
	builder.CreateBr(loopBB);

	builder.SetInsertPoint(loopBB);
	auto yPhi = builder.CreatePHI(_type, 2, "y.phi");
	auto rPhi = builder.CreatePHI(_type, 2, "r.phi");
	auto iPhi = builder.CreatePHI(_type, 2, "i.phi");
	auto qPhi = builder.CreatePHI(_type, 2, "q.phi");
	auto rUpdate = builder.CreateNUWSub(rPhi, yPhi);
	auto qUpdate = builder.CreateOr(qPhi, one);	// q += 1, q lowest bit is 0
	auto rGEy = builder.CreateICmpUGE(rPhi, yPhi);
	auto r1 = builder.CreateSelect(rGEy, rUpdate, rPhi, "r1");
	auto q1 = builder.CreateSelect(rGEy, qUpdate, qPhi, "q");
	auto iZero = builder.CreateICmpEQ(iPhi, zero);
	builder.CreateCondBr(iZero, returnBB, continueBB);

	builder.SetInsertPoint(continueBB);
	auto i2 = builder.CreateNUWSub(iPhi, one);
	auto q2 = builder.CreateShl(q1, one);
	auto y2 = builder.CreateLShr(yPhi, one);
	builder.CreateBr(loopBB);

	yPhi->addIncoming(y0, mainBB);
	yPhi->addIncoming(y2, continueBB);
	rPhi->addIncoming(r0, mainBB);
	rPhi->addIncoming(r1, continueBB);
	iPhi->addIncoming(i0, mainBB);
	iPhi->addIncoming(i2, continueBB);
	qPhi->addIncoming(zero, mainBB);
	qPhi->addIncoming(q2, continueBB);

	builder.SetInsertPoint(returnBB);
	auto qRet = builder.CreatePHI(_type, 2, "q.ret");
	qRet->addIncoming(zero, entryBB);
	qRet->addIncoming(q1, loopBB);
	auto rRet = builder.CreatePHI(_type, 2, "r.ret");
	rRet->addIncoming(r0, entryBB);
	rRet->addIncoming(r1, loopBB);
	auto ret = builder.CreateInsertElement(llvm::UndefValue::get(retType), qRet, uint64_t(0), "ret0");
	ret = builder.CreateInsertElement(ret, rRet, 1, "ret");
	builder.CreateRet(ret);

	return func;
}
}

llvm::Function* Arith256::getUDivRem256Func(llvm::Module& _module)
{
	static const auto funcName = "evm.udivrem.i256";
	if (auto func = _module.getFunction(funcName))
		return func;

	return createUDivRemFunc(Type::Word, _module, funcName);
}

llvm::Function* Arith256::getUDivRem512Func(llvm::Module& _module)
{
	static const auto funcName = "evm.udivrem.i512";
	if (auto func = _module.getFunction(funcName))
		return func;

	return createUDivRemFunc(llvm::IntegerType::get(_module.getContext(), 512), _module, funcName);
}

llvm::Function* Arith256::getUDiv256Func(llvm::Module& _module)
{
	static const auto funcName = "evm.udiv.i256";
	if (auto func = _module.getFunction(funcName))
		return func;

	auto udivremFunc = getUDivRem256Func(_module);

	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Word, {Type::Word, Type::Word}, false), llvm::Function::PrivateLinkage, funcName, &_module);
	func->setDoesNotThrow();
	func->setDoesNotAccessMemory();

	auto x = &func->getArgumentList().front();
	x->setName("x");
	auto y = x->getNextNode();
	y->setName("y");

	auto bb = llvm::BasicBlock::Create(_module.getContext(), {}, func);
	auto builder = llvm::IRBuilder<>{bb};
	auto udivrem = builder.CreateCall(udivremFunc, {x, y});
	auto udiv = builder.CreateExtractElement(udivrem, uint64_t(0));
	builder.CreateRet(udiv);

	return func;
}

namespace
{
llvm::Function* createURemFunc(llvm::Type* _type, llvm::Module& _module, char const* _funcName)
{
	auto udivremFunc = _type == Type::Word ? Arith256::getUDivRem256Func(_module) : Arith256::getUDivRem512Func(_module);

	auto func = llvm::Function::Create(llvm::FunctionType::get(_type, {_type, _type}, false), llvm::Function::PrivateLinkage, _funcName, &_module);
	func->setDoesNotThrow();
	func->setDoesNotAccessMemory();

	auto x = &func->getArgumentList().front();
	x->setName("x");
	auto y = x->getNextNode();
	y->setName("y");

	auto bb = llvm::BasicBlock::Create(_module.getContext(), {}, func);
	auto builder = llvm::IRBuilder<>{bb};
	auto udivrem = builder.CreateCall(udivremFunc, {x, y});
	auto r = builder.CreateExtractElement(udivrem, uint64_t(1));
	builder.CreateRet(r);

	return func;
}
}

llvm::Function* Arith256::getURem256Func(llvm::Module& _module)
{
	static const auto funcName = "evm.urem.i256";
	if (auto func = _module.getFunction(funcName))
		return func;
	return createURemFunc(Type::Word, _module, funcName);
}

llvm::Function* Arith256::getURem512Func(llvm::Module& _module)
{
	static const auto funcName = "evm.urem.i512";
	if (auto func = _module.getFunction(funcName))
		return func;
	return createURemFunc(llvm::IntegerType::get(_module.getContext(), 512), _module, funcName);
}

llvm::Function* Arith256::getSDivRem256Func(llvm::Module& _module)
{
	static const auto funcName = "evm.sdivrem.i256";
	if (auto func = _module.getFunction(funcName))
		return func;

	auto udivremFunc = getUDivRem256Func(_module);

	auto retType = llvm::VectorType::get(Type::Word, 2);
	auto func = llvm::Function::Create(llvm::FunctionType::get(retType, {Type::Word, Type::Word}, false), llvm::Function::PrivateLinkage, funcName, &_module);
	func->setDoesNotThrow();
	func->setDoesNotAccessMemory();

	auto x = &func->getArgumentList().front();
	x->setName("x");
	auto y = x->getNextNode();
	y->setName("y");

	auto bb = llvm::BasicBlock::Create(_module.getContext(), "", func);
	auto builder = llvm::IRBuilder<>{bb};
	auto xIsNeg = builder.CreateICmpSLT(x, Constant::get(0));
	auto xNeg = builder.CreateSub(Constant::get(0), x);
	auto xAbs = builder.CreateSelect(xIsNeg, xNeg, x);

	auto yIsNeg = builder.CreateICmpSLT(y, Constant::get(0));
	auto yNeg = builder.CreateSub(Constant::get(0), y);
	auto yAbs = builder.CreateSelect(yIsNeg, yNeg, y);

	auto res = builder.CreateCall(udivremFunc, {xAbs, yAbs});
	auto qAbs = builder.CreateExtractElement(res, uint64_t(0));
	auto rAbs = builder.CreateExtractElement(res, 1);

	// the reminder has the same sign as dividend
	auto rNeg = builder.CreateSub(Constant::get(0), rAbs);
	auto r = builder.CreateSelect(xIsNeg, rNeg, rAbs);

	auto qNeg = builder.CreateSub(Constant::get(0), qAbs);
	auto xyOpposite = builder.CreateXor(xIsNeg, yIsNeg);
	auto q = builder.CreateSelect(xyOpposite, qNeg, qAbs);

	auto ret = builder.CreateInsertElement(llvm::UndefValue::get(retType), q, uint64_t(0));
	ret = builder.CreateInsertElement(ret, r, 1);
	builder.CreateRet(ret);

	return func;
}

llvm::Function* Arith256::getSDiv256Func(llvm::Module& _module)
{
	static const auto funcName = "evm.sdiv.i256";
	if (auto func = _module.getFunction(funcName))
		return func;

	auto sdivremFunc = getSDivRem256Func(_module);

	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Word, {Type::Word, Type::Word}, false), llvm::Function::PrivateLinkage, funcName, &_module);
	func->setDoesNotThrow();
	func->setDoesNotAccessMemory();

	auto x = &func->getArgumentList().front();
	x->setName("x");
	auto y = x->getNextNode();
	y->setName("y");

	auto bb = llvm::BasicBlock::Create(_module.getContext(), {}, func);
	auto builder = llvm::IRBuilder<>{bb};
	auto sdivrem = builder.CreateCall(sdivremFunc, {x, y});
	auto q = builder.CreateExtractElement(sdivrem, uint64_t(0));
	builder.CreateRet(q);

	return func;
}

llvm::Function* Arith256::getSRem256Func(llvm::Module& _module)
{
	static const auto funcName = "evm.srem.i256";
	if (auto func = _module.getFunction(funcName))
		return func;

	auto sdivremFunc = getSDivRem256Func(_module);

	auto func = llvm::Function::Create(llvm::FunctionType::get(Type::Word, {Type::Word, Type::Word}, false), llvm::Function::PrivateLinkage, funcName, &_module);
	func->setDoesNotThrow();
	func->setDoesNotAccessMemory();

	auto x = &func->getArgumentList().front();
	x->setName("x");
	auto y = x->getNextNode();
	y->setName("y");

	auto bb = llvm::BasicBlock::Create(_module.getContext(), {}, func);
	auto builder = llvm::IRBuilder<>{bb};
	auto sdivrem = builder.CreateCall(sdivremFunc, {x, y});
	auto r = builder.CreateExtractElement(sdivrem, uint64_t(1));
	builder.CreateRet(r);

	return func;
}

llvm::Function* Arith256::getExpFunc()
{
	if (!m_exp)
	{
		llvm::Type* argTypes[] = {Type::Word, Type::Word};
		m_exp = llvm::Function::Create(llvm::FunctionType::get(Type::Word, argTypes, false), llvm::Function::PrivateLinkage, "exp", getModule());
		m_exp->setDoesNotThrow();
		m_exp->setDoesNotAccessMemory();

		auto base = &m_exp->getArgumentList().front();
		base->setName("base");
		auto exponent = base->getNextNode();
		exponent->setName("exponent");

		InsertPointGuard guard{m_builder};

		//	while (e != 0) {
		//		if (e % 2 == 1)
		//			r *= b;
		//		b *= b;
		//		e /= 2;
		//	}

		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), "Entry", m_exp);
		auto headerBB = llvm::BasicBlock::Create(m_builder.getContext(), "LoopHeader", m_exp);
		auto bodyBB = llvm::BasicBlock::Create(m_builder.getContext(), "LoopBody", m_exp);
		auto updateBB = llvm::BasicBlock::Create(m_builder.getContext(), "ResultUpdate", m_exp);
		auto continueBB = llvm::BasicBlock::Create(m_builder.getContext(), "Continue", m_exp);
		auto returnBB = llvm::BasicBlock::Create(m_builder.getContext(), "Return", m_exp);

		m_builder.SetInsertPoint(entryBB);
		m_builder.CreateBr(headerBB);

		m_builder.SetInsertPoint(headerBB);
		auto r = m_builder.CreatePHI(Type::Word, 2, "r");
		auto b = m_builder.CreatePHI(Type::Word, 2, "b");
		auto e = m_builder.CreatePHI(Type::Word, 2, "e");
		auto eNonZero = m_builder.CreateICmpNE(e, Constant::get(0), "e.nonzero");
		m_builder.CreateCondBr(eNonZero, bodyBB, returnBB);

		m_builder.SetInsertPoint(bodyBB);
		auto eOdd = m_builder.CreateICmpNE(m_builder.CreateAnd(e, Constant::get(1)), Constant::get(0), "e.isodd");
		m_builder.CreateCondBr(eOdd, updateBB, continueBB);

		m_builder.SetInsertPoint(updateBB);
		auto mul256Func = getMulFunc(*getModule());
		auto r0 = createCall(mul256Func, {r, b});
		m_builder.CreateBr(continueBB);

		m_builder.SetInsertPoint(continueBB);
		auto r1 = m_builder.CreatePHI(Type::Word, 2, "r1");
		r1->addIncoming(r, bodyBB);
		r1->addIncoming(r0, updateBB);
		auto b1 = createCall(mul256Func, {b, b});
		auto e1 = m_builder.CreateLShr(e, Constant::get(1), "e1");
		m_builder.CreateBr(headerBB);

		r->addIncoming(Constant::get(1), entryBB);
		r->addIncoming(r1, continueBB);
		b->addIncoming(base, entryBB);
		b->addIncoming(b1, continueBB);
		e->addIncoming(exponent, entryBB);
		e->addIncoming(e1, continueBB);

		m_builder.SetInsertPoint(returnBB);
		m_builder.CreateRet(r);
	}
	return m_exp;
}

llvm::Value* Arith256::exp(llvm::Value* _arg1, llvm::Value* _arg2)
{
	//	while (e != 0) {
	//		if (e % 2 == 1)
	//			r *= b;
	//		b *= b;
	//		e /= 2;
	//	}

	if (auto c1 = llvm::dyn_cast<llvm::ConstantInt>(_arg1))
	{
		if (auto c2 = llvm::dyn_cast<llvm::ConstantInt>(_arg2))
		{
			auto b = c1->getValue();
			auto e = c2->getValue();
			auto r = llvm::APInt{256, 1};
			while (e != 0)
			{
				if (e[0])
					r *= b;
				b *= b;
				e = e.lshr(1);
			}
			return Constant::get(r);
		}
	}

	return createCall(getExpFunc(), {_arg1, _arg2});
}

}
}
}

extern "C"
{
	EXPORT void debug(uint64_t a, uint64_t b, uint64_t c, uint64_t d, char z)
	{
		DLOG(JIT) << "DEBUG " << std::dec << z << ": " //<< d << c << b << a
				<< " ["	<< std::hex << std::setfill('0') << std::setw(16) << d << std::setw(16) << c << std::setw(16) << b << std::setw(16) << a << "]\n";
	}
}
