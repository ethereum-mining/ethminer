#include "Arith256.h"

#include <iostream>
#include <iomanip>

#include "preprocessor/llvm_includes_start.h"
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

llvm::Function* Arith256::getMulFunc()
{
	auto& func = m_mul;
	if (!func)
	{
		llvm::Type* argTypes[] = {Type::Word, Type::Word};
		func = llvm::Function::Create(llvm::FunctionType::get(Type::Word, argTypes, false), llvm::Function::PrivateLinkage, "mul", getModule());
		func->setDoesNotThrow();
		func->setDoesNotAccessMemory();

		auto x = &func->getArgumentList().front();
		x->setName("x");
		auto y = x->getNextNode();
		y->setName("y");

		InsertPointGuard guard{m_builder};
		auto bb = llvm::BasicBlock::Create(m_builder.getContext(), {}, func);
		m_builder.SetInsertPoint(bb);
		auto i64 = Type::Size;
		auto i128 = m_builder.getIntNTy(128);
		auto i256 = Type::Word;
		auto c64 = Constant::get(64);
		auto c128 = Constant::get(128);
		auto c192 = Constant::get(192);

		auto x_lo = m_builder.CreateTrunc(x, i64, "x.lo");
		auto y_lo = m_builder.CreateTrunc(y, i64, "y.lo");
		auto x_mi = m_builder.CreateTrunc(m_builder.CreateLShr(x, c64), i64);
		auto y_mi = m_builder.CreateTrunc(m_builder.CreateLShr(y, c64), i64);
		auto x_hi = m_builder.CreateTrunc(m_builder.CreateLShr(x, c128), i128);
		auto y_hi = m_builder.CreateTrunc(m_builder.CreateLShr(y, c128), i128);

		auto t1 = m_builder.CreateMul(m_builder.CreateZExt(x_lo, i128), m_builder.CreateZExt(y_lo, i128));
		auto t2 = m_builder.CreateMul(m_builder.CreateZExt(x_lo, i128), m_builder.CreateZExt(y_mi, i128));
		auto t3 = m_builder.CreateMul(m_builder.CreateZExt(x_lo, i128), y_hi);
		auto t4 = m_builder.CreateMul(m_builder.CreateZExt(x_mi, i128), m_builder.CreateZExt(y_lo, i128));
		auto t5 = m_builder.CreateMul(m_builder.CreateZExt(x_mi, i128), m_builder.CreateZExt(y_mi, i128));
		auto t6 = m_builder.CreateMul(m_builder.CreateZExt(x_mi, i128), y_hi);
		auto t7 = m_builder.CreateMul(x_hi, m_builder.CreateZExt(y_lo, i128));
		auto t8 = m_builder.CreateMul(x_hi, m_builder.CreateZExt(y_mi, i128));

		auto p = m_builder.CreateZExt(t1, i256);
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t2, i256), c64));
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t3, i256), c128));
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t4, i256), c64));
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t5, i256), c128));
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t6, i256), c192));
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t7, i256), c128));
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t8, i256), c192));
		m_builder.CreateRet(p);
	}
	return func;
}

llvm::Function* Arith256::getMul512Func()
{
	auto& func = m_mul512;
	if (!func)
	{
		auto i512 = m_builder.getIntNTy(512);
		llvm::Type* argTypes[] = {Type::Word, Type::Word};
		func = llvm::Function::Create(llvm::FunctionType::get(i512, argTypes, false), llvm::Function::PrivateLinkage, "mul512", getModule());
		func->setDoesNotThrow();
		func->setDoesNotAccessMemory();

		auto x = &func->getArgumentList().front();
		x->setName("x");
		auto y = x->getNextNode();
		y->setName("y");

		InsertPointGuard guard{m_builder};
		auto bb = llvm::BasicBlock::Create(m_builder.getContext(), {}, func);
		m_builder.SetInsertPoint(bb);
		auto i128 = m_builder.getIntNTy(128);
		auto i256 = Type::Word;
		auto x_lo = m_builder.CreateZExt(m_builder.CreateTrunc(x, i128, "x.lo"), i256);
		auto y_lo = m_builder.CreateZExt(m_builder.CreateTrunc(y, i128, "y.lo"), i256);
		auto x_hi = m_builder.CreateZExt(m_builder.CreateTrunc(m_builder.CreateLShr(x, Constant::get(128)), i128, "x.hi"), i256);
		auto y_hi = m_builder.CreateZExt(m_builder.CreateTrunc(m_builder.CreateLShr(y, Constant::get(128)), i128, "y.hi"), i256);

		auto t1 = createCall(getMulFunc(), {x_lo, y_lo});
		auto t2 = createCall(getMulFunc(), {x_lo, y_hi});
		auto t3 = createCall(getMulFunc(), {x_hi, y_lo});
		auto t4 = createCall(getMulFunc(), {x_hi, y_hi});

		auto p = m_builder.CreateZExt(t1, i512);
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t2, i512), m_builder.getIntN(512, 128)));
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t3, i512), m_builder.getIntN(512, 128)));
		p = m_builder.CreateAdd(p, m_builder.CreateShl(m_builder.CreateZExt(t4, i512), m_builder.getIntN(512, 256)));
		m_builder.CreateRet(p);
	}
	return func;
}

llvm::Function* Arith256::getDivFunc(llvm::Type* _type)
{
	auto& func = _type == Type::Word ? m_div : m_div512;

	if (!func)
	{
		// Based of "Improved shift divisor algorithm" from "Software Integer Division" by Microsoft Research
		// The following algorithm also handles divisor of value 0 returning 0 for both quotient and reminder

		llvm::Type* argTypes[] = {_type, _type};
		auto retType = llvm::StructType::get(m_builder.getContext(), llvm::ArrayRef<llvm::Type*>{argTypes});
		auto funcName = _type == Type::Word ? "div" : "div512";
		func = llvm::Function::Create(llvm::FunctionType::get(retType, argTypes, false), llvm::Function::PrivateLinkage, funcName, getModule());
		func->setDoesNotThrow();
		func->setDoesNotAccessMemory();

		auto zero = llvm::ConstantInt::get(_type, 0);
		auto one = llvm::ConstantInt::get(_type, 1);

		auto x = &func->getArgumentList().front();
		x->setName("x");
		auto yArg = x->getNextNode();
		yArg->setName("y");

		InsertPointGuard guard{m_builder};

		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), "Entry", func);
		auto mainBB = llvm::BasicBlock::Create(m_builder.getContext(), "Main", func);
		auto loopBB = llvm::BasicBlock::Create(m_builder.getContext(), "Loop", func);
		auto continueBB = llvm::BasicBlock::Create(m_builder.getContext(), "Continue", func);
		auto returnBB = llvm::BasicBlock::Create(m_builder.getContext(), "Return", func);

		m_builder.SetInsertPoint(entryBB);
		auto yNonZero = m_builder.CreateICmpNE(yArg, zero);
		auto yLEx = m_builder.CreateICmpULE(yArg, x);
		auto r0 = m_builder.CreateSelect(yNonZero, x, zero, "r0");
		m_builder.CreateCondBr(m_builder.CreateAnd(yLEx, yNonZero), mainBB, returnBB);

		m_builder.SetInsertPoint(mainBB);
		auto ctlzIntr = llvm::Intrinsic::getDeclaration(getModule(), llvm::Intrinsic::ctlz, _type);
		// both y and r are non-zero
		auto yLz = m_builder.CreateCall(ctlzIntr, {yArg, m_builder.getInt1(true)}, "y.lz");
		auto rLz = m_builder.CreateCall(ctlzIntr, {r0, m_builder.getInt1(true)}, "r.lz");
		auto i0 = m_builder.CreateNUWSub(yLz, rLz, "i0");
		auto y0 = m_builder.CreateShl(yArg, i0);
		m_builder.CreateBr(loopBB);

		m_builder.SetInsertPoint(loopBB);
		auto yPhi = m_builder.CreatePHI(_type, 2, "y.phi");
		auto rPhi = m_builder.CreatePHI(_type, 2, "r.phi");
		auto iPhi = m_builder.CreatePHI(_type, 2, "i.phi");
		auto qPhi = m_builder.CreatePHI(_type, 2, "q.phi");
		auto rUpdate = m_builder.CreateNUWSub(rPhi, yPhi);
		auto qUpdate = m_builder.CreateOr(qPhi, one);	// q += 1, q lowest bit is 0
		auto rGEy = m_builder.CreateICmpUGE(rPhi, yPhi);
		auto r1 = m_builder.CreateSelect(rGEy, rUpdate, rPhi, "r1");
		auto q1 = m_builder.CreateSelect(rGEy, qUpdate, qPhi, "q");
		auto iZero = m_builder.CreateICmpEQ(iPhi, zero);
		m_builder.CreateCondBr(iZero, returnBB, continueBB);

		m_builder.SetInsertPoint(continueBB);
		auto i2 = m_builder.CreateNUWSub(iPhi, one);
		auto q2 = m_builder.CreateShl(q1, one);
		auto y2 = m_builder.CreateLShr(yPhi, one);
		m_builder.CreateBr(loopBB);

		yPhi->addIncoming(y0, mainBB);
		yPhi->addIncoming(y2, continueBB);
		rPhi->addIncoming(r0, mainBB);
		rPhi->addIncoming(r1, continueBB);
		iPhi->addIncoming(i0, mainBB);
		iPhi->addIncoming(i2, continueBB);
		qPhi->addIncoming(zero, mainBB);
		qPhi->addIncoming(q2, continueBB);

		m_builder.SetInsertPoint(returnBB);
		auto qRet = m_builder.CreatePHI(_type, 2, "q.ret");
		qRet->addIncoming(zero, entryBB);
		qRet->addIncoming(q1, loopBB);
		auto rRet = m_builder.CreatePHI(_type, 2, "r.ret");
		rRet->addIncoming(r0, entryBB);
		rRet->addIncoming(r1, loopBB);
		auto ret = m_builder.CreateInsertValue(llvm::UndefValue::get(retType), qRet, 0, "ret0");
		ret = m_builder.CreateInsertValue(ret, rRet, 1, "ret");
		m_builder.CreateRet(ret);
	}
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
		auto r0 = createCall(getMulFunc(), {r, b});
		m_builder.CreateBr(continueBB);

		m_builder.SetInsertPoint(continueBB);
		auto r1 = m_builder.CreatePHI(Type::Word, 2, "r1");
		r1->addIncoming(r, bodyBB);
		r1->addIncoming(r0, updateBB);
		auto b1 = createCall(getMulFunc(), {b, b});
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

llvm::Function* Arith256::getAddModFunc()
{
	if (!m_addmod)
	{
		auto i512Ty = m_builder.getIntNTy(512);
		llvm::Type* argTypes[] = {Type::Word, Type::Word, Type::Word};
		m_addmod = llvm::Function::Create(llvm::FunctionType::get(Type::Word, argTypes, false), llvm::Function::PrivateLinkage, "addmod", getModule());
		m_addmod->setDoesNotThrow();
		m_addmod->setDoesNotAccessMemory();

		auto x = &m_addmod->getArgumentList().front();
		x->setName("x");
		auto y = x->getNextNode();
		y->setName("y");
		auto mod = y->getNextNode();
		mod->setName("m");

		InsertPointGuard guard{m_builder};

		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, m_addmod);
		m_builder.SetInsertPoint(entryBB);
		auto x512 = m_builder.CreateZExt(x, i512Ty, "x512");
		auto y512 = m_builder.CreateZExt(y, i512Ty, "y512");
		auto m512 = m_builder.CreateZExt(mod, i512Ty, "m512");
		auto s = m_builder.CreateAdd(x512, y512, "s");
		auto d = createCall(getDivFunc(i512Ty), {s, m512});
		auto r = m_builder.CreateExtractValue(d, 1, "r");
		m_builder.CreateRet(m_builder.CreateTrunc(r, Type::Word));
	}
	return m_addmod;
}

llvm::Function* Arith256::getMulModFunc()
{
	if (!m_mulmod)
	{
		llvm::Type* argTypes[] = {Type::Word, Type::Word, Type::Word};
		m_mulmod = llvm::Function::Create(llvm::FunctionType::get(Type::Word, argTypes, false), llvm::Function::PrivateLinkage, "mulmod", getModule());
		m_mulmod->setDoesNotThrow();
		m_mulmod->setDoesNotAccessMemory();

		auto i512Ty = m_builder.getIntNTy(512);
		auto x = &m_mulmod->getArgumentList().front();
		x->setName("x");
		auto y = x->getNextNode();
		y->setName("y");
		auto mod = y->getNextNode();
		mod->setName("mod");

		InsertPointGuard guard{m_builder};

		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, m_mulmod);
		m_builder.SetInsertPoint(entryBB);
		auto p = createCall(getMul512Func(), {x, y});
		auto m = m_builder.CreateZExt(mod, i512Ty, "m");
		auto d = createCall(getDivFunc(i512Ty), {p, m});
		auto r = m_builder.CreateExtractValue(d, 1, "r");
		r = m_builder.CreateTrunc(r, Type::Word);
		m_builder.CreateRet(r);
	}
	return m_mulmod;
}

llvm::Value* Arith256::mul(llvm::Value* _arg1, llvm::Value* _arg2)
{
	if (auto c1 = llvm::dyn_cast<llvm::ConstantInt>(_arg1))
	{
		if (auto c2 = llvm::dyn_cast<llvm::ConstantInt>(_arg2))
			return Constant::get(c1->getValue() * c2->getValue());
	}

	return createCall(getMulFunc(), {_arg1, _arg2});
}

std::pair<llvm::Value*, llvm::Value*> Arith256::div(llvm::Value* _arg1, llvm::Value* _arg2)
{
	// FIXME: Disabled because of llvm::APInt::urem bug
//	if (auto c1 = llvm::dyn_cast<llvm::ConstantInt>(_arg1))
//	{
//		if (auto c2 = llvm::dyn_cast<llvm::ConstantInt>(_arg2))
//		{
//			if (!c2->getValue())
//				return std::make_pair(Constant::get(0), Constant::get(0));
//			auto div = Constant::get(c1->getValue().udiv(c2->getValue()));
//			auto mod = Constant::get(c1->getValue().urem(c2->getValue()));
//			return std::make_pair(div, mod);
//		}
//	}

	auto r = createCall(getDivFunc(Type::Word), {_arg1, _arg2});
	auto div =  m_builder.CreateExtractValue(r, 0, "div");
	auto mod =  m_builder.CreateExtractValue(r, 1, "mod");
	return std::make_pair(div, mod);
}

std::pair<llvm::Value*, llvm::Value*> Arith256::sdiv(llvm::Value* _x, llvm::Value* _y)
{
	// FIXME: Disabled because of llvm::APInt::urem bug
//	if (auto c1 = llvm::dyn_cast<llvm::ConstantInt>(_x))
//	{
//		if (auto c2 = llvm::dyn_cast<llvm::ConstantInt>(_y))
//		{
//			if (!c2->getValue())
//				return std::make_pair(Constant::get(0), Constant::get(0));
//			auto div = Constant::get(c1->getValue().sdiv(c2->getValue()));
//			auto mod = Constant::get(c1->getValue().srem(c2->getValue()));
//			return std::make_pair(div, mod);
//		}
//	}

	auto xIsNeg = m_builder.CreateICmpSLT(_x, Constant::get(0));
	auto xNeg = m_builder.CreateSub(Constant::get(0), _x);
	auto xAbs = m_builder.CreateSelect(xIsNeg, xNeg, _x);

	auto yIsNeg = m_builder.CreateICmpSLT(_y, Constant::get(0));
	auto yNeg = m_builder.CreateSub(Constant::get(0), _y);
	auto yAbs = m_builder.CreateSelect(yIsNeg, yNeg, _y);

	auto res = div(xAbs, yAbs);

	// the reminder has the same sign as dividend
	auto rNeg = m_builder.CreateSub(Constant::get(0), res.second);
	res.second = m_builder.CreateSelect(xIsNeg, rNeg, res.second);

	auto qNeg = m_builder.CreateSub(Constant::get(0), res.first);
	auto xyOpposite = m_builder.CreateXor(xIsNeg, yIsNeg);
	res.first = m_builder.CreateSelect(xyOpposite, qNeg, res.first);

	return res;
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

llvm::Value* Arith256::addmod(llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3)
{
	// FIXME: Disabled because of llvm::APInt::urem bug
//	if (auto c1 = llvm::dyn_cast<llvm::ConstantInt>(_arg1))
//	{
//		if (auto c2 = llvm::dyn_cast<llvm::ConstantInt>(_arg2))
//		{
//			if (auto c3 = llvm::dyn_cast<llvm::ConstantInt>(_arg3))
//			{
//				if (!c3->getValue())
//					return Constant::get(0);
//				auto s = c1->getValue().zext(256+64) + c2->getValue().zext(256+64);
//				auto r = s.urem(c3->getValue().zext(256+64)).trunc(256);
//				return Constant::get(r);
//			}
//		}
//	}

	return createCall(getAddModFunc(), {_arg1, _arg2, _arg3});
}

llvm::Value* Arith256::mulmod(llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3)
{
	// FIXME: Disabled because of llvm::APInt::urem bug
//	if (auto c1 = llvm::dyn_cast<llvm::ConstantInt>(_arg1))
//	{
//		if (auto c2 = llvm::dyn_cast<llvm::ConstantInt>(_arg2))
//		{
//			if (auto c3 = llvm::dyn_cast<llvm::ConstantInt>(_arg3))
//			{
//				if (!c3->getValue())
//					return Constant::get(0);
//				auto p = c1->getValue().zext(512) * c2->getValue().zext(512);
//				auto r = p.urem(c3->getValue().zext(512)).trunc(256);
//				return Constant::get(r);
//			}
//		}
//	}

	return createCall(getMulModFunc(), {_arg1, _arg2, _arg3});
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
