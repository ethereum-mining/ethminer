#include "Arith256.h"
#include "Runtime.h"
#include "Type.h"
#include "Endianness.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IntrinsicInst.h>
#include <gmp.h>
#include <iostream>
#include <iomanip>

namespace dev
{
namespace eth
{
namespace jit
{

Arith256::Arith256(llvm::IRBuilder<>& _builder) :
	CompilerHelper(_builder)
{
	using namespace llvm;

	m_result = m_builder.CreateAlloca(Type::Word, nullptr, "arith.result");
	m_arg1 = m_builder.CreateAlloca(Type::Word, nullptr, "arith.arg1");
	m_arg2 = m_builder.CreateAlloca(Type::Word, nullptr, "arith.arg2");
	m_arg3 = m_builder.CreateAlloca(Type::Word, nullptr, "arith.arg3");

	using Linkage = GlobalValue::LinkageTypes;

	llvm::Type* arg2Types[] = {Type::WordPtr, Type::WordPtr, Type::WordPtr};
	llvm::Type* arg3Types[] = {Type::WordPtr, Type::WordPtr, Type::WordPtr, Type::WordPtr};

	m_mul = Function::Create(FunctionType::get(Type::Void, arg2Types, false), Linkage::ExternalLinkage, "arith_mul", getModule());
	m_addmod = Function::Create(FunctionType::get(Type::Void, arg3Types, false), Linkage::ExternalLinkage, "arith_addmod", getModule());
}

void Arith256::debug(llvm::Value* _value, char _c)
{
	if (!m_debug)
	{
		llvm::Type* argTypes[] = {Type::Word, m_builder.getInt8Ty()};
		m_debug = llvm::Function::Create(llvm::FunctionType::get(Type::Void, argTypes, false), llvm::Function::ExternalLinkage, "debug", getModule());
	}
	createCall(m_debug, {_value, m_builder.getInt8(_c)});
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
		auto yLz = m_builder.CreateCall2(ctlzIntr, yArg, m_builder.getInt1(true), "y.lz");
		auto rLz = m_builder.CreateCall2(ctlzIntr, r0, m_builder.getInt1(true), "r.lz");
		auto i0 = m_builder.CreateNUWSub(yLz, rLz, "i0");
		auto shlBy0 = m_builder.CreateICmpEQ(i0, zero);
		auto y0 = m_builder.CreateShl(yArg, i0);
		y0 = m_builder.CreateSelect(shlBy0, yArg, y0, "y0"); // Workaround for LLVM bug: shl by 0 produces wrong result
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
		m_exp = llvm::Function::Create(llvm::FunctionType::get(Type::Word, argTypes, false), llvm::Function::PrivateLinkage, "arith.exp", getModule());

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
		auto a1 = m_builder.CreateAlloca(Type::Word, nullptr, "a1");
		auto a2 = m_builder.CreateAlloca(Type::Word, nullptr, "a2");
		auto a3 = m_builder.CreateAlloca(Type::Word, nullptr, "a3");
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
		m_builder.CreateStore(r, a1);
		m_builder.CreateStore(b, a2);
		createCall(m_mul, {a1, a2, a3});
		auto r0 = m_builder.CreateLoad(a3, "r0");
		m_builder.CreateBr(continueBB);

		m_builder.SetInsertPoint(continueBB);
		auto r1 = m_builder.CreatePHI(Type::Word, 2, "r1");
		r1->addIncoming(r, bodyBB);
		r1->addIncoming(r0, updateBB);
		m_builder.CreateStore(b, a1);
		m_builder.CreateStore(b, a2);
		createCall(m_mul, {a1, a2, a3});
		auto b1 = m_builder.CreateLoad(a3, "b1");
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

llvm::Function* Arith256::getMulModFunc()
{
	if (!m_mulmod)
	{
		llvm::Type* argTypes[] = {Type::Word, Type::Word, Type::Word};
		m_mulmod = llvm::Function::Create(llvm::FunctionType::get(Type::Word, argTypes, false), llvm::Function::PrivateLinkage, "mulmod", getModule());

		auto i512Ty = m_builder.getIntNTy(512);
		llvm::Type* mul512ArgTypes[] = {Type::WordPtr, Type::WordPtr, i512Ty->getPointerTo()};
		auto mul512 = llvm::Function::Create(llvm::FunctionType::get(Type::Void, mul512ArgTypes, false), llvm::Function::ExternalLinkage, "arith_mul512", getModule());

		auto x = &m_mulmod->getArgumentList().front();
		x->setName("x");
		auto y = x->getNextNode();
		y->setName("y");
		auto mod = y->getNextNode();
		mod->setName("mod");

		InsertPointGuard guard{m_builder};

		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), {}, m_mulmod);
		m_builder.SetInsertPoint(entryBB);
		auto a1 = m_builder.CreateAlloca(Type::Word);
		auto a2 = m_builder.CreateAlloca(Type::Word);
		auto a3 = m_builder.CreateAlloca(i512Ty);
		m_builder.CreateStore(x, a1);
		m_builder.CreateStore(y, a2);
		createCall(mul512, {a1, a2, a3});
		auto p = m_builder.CreateLoad(a3, "p");
		auto m = m_builder.CreateZExt(mod, i512Ty, "m");
		auto d = createCall(getDivFunc(i512Ty), {p, m});
		auto r = m_builder.CreateExtractValue(d, 1, "r");
		m_builder.CreateRet(r);
	}
	return m_mulmod;
}


llvm::Value* Arith256::binaryOp(llvm::Function* _op, llvm::Value* _arg1, llvm::Value* _arg2)
{
	m_builder.CreateStore(_arg1, m_arg1);
	m_builder.CreateStore(_arg2, m_arg2);
	m_builder.CreateCall3(_op, m_arg1, m_arg2, m_result);
	return m_builder.CreateLoad(m_result);
}

llvm::Value* Arith256::ternaryOp(llvm::Function* _op, llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3)
{
	m_builder.CreateStore(_arg1, m_arg1);
	m_builder.CreateStore(_arg2, m_arg2);
	m_builder.CreateStore(_arg3, m_arg3);
	m_builder.CreateCall4(_op, m_arg1, m_arg2, m_arg3, m_result);
	return m_builder.CreateLoad(m_result);
}

llvm::Value* Arith256::mul(llvm::Value* _arg1, llvm::Value* _arg2)
{
	return binaryOp(m_mul, _arg1, _arg2);
}

std::pair<llvm::Value*, llvm::Value*> Arith256::div(llvm::Value* _arg1, llvm::Value* _arg2)
{
	auto div =  m_builder.CreateExtractValue(createCall(getDivFunc(Type::Word), {_arg1, _arg2}), 0, "div");
	auto mod =  m_builder.CreateExtractValue(createCall(getDivFunc(Type::Word), {_arg1, _arg2}), 1, "mod");
	return std::make_pair(div, mod);
}

std::pair<llvm::Value*, llvm::Value*> Arith256::sdiv(llvm::Value* _x, llvm::Value* _y)
{
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
	return createCall(getExpFunc(), {_arg1, _arg2});
}

llvm::Value* Arith256::addmod(llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3)
{
	return ternaryOp(m_addmod, _arg1, _arg2, _arg3);
}

llvm::Value* Arith256::mulmod(llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3)
{
	return createCall(getMulModFunc(), {_arg1, _arg2, _arg3});
}

namespace
{
	using uint128 = __uint128_t;

	struct uint256
	{
		uint64_t lo = 0;
		uint64_t mid = 0;
		uint128 hi = 0;

		uint256(uint64_t lo, uint64_t mid, uint128 hi): lo(lo), mid(mid), hi(hi) {}
		uint256(uint128 n)
		{
			*((uint128*)&lo) = n;
		}

		explicit operator uint128()
		{
			return *((uint128*)&lo);
		}

		uint256 operator|(uint256 a)
		{
			return {lo | a.lo, mid | a.mid, hi | a.hi};
		}

		uint256 operator+(uint256 a)
		{
			auto _lo = (uint128) lo + a.lo;
			auto _mid = (uint128) mid + a.mid + (_lo >> 64);
			auto _hi = hi + a.hi + (_mid >> 64);
			return {(uint64_t)_lo, (uint64_t)_mid, _hi};
		}

		uint256 lo2hi()
		{
			hi = (uint128)*this;
			lo = 0;
			mid = 0;
			return *this;
		}
	};

	struct uint512
	{
		uint128 lo;
		uint128 mid;
		uint256 hi;
	};

	uint256 mul(uint256 x, uint256 y)
	{
		auto t1 = (uint128) x.lo * y.lo;
		auto t2 = (uint128) x.lo * y.mid;
		auto t3 = x.lo * y.hi;
		auto t4 = (uint128) x.mid * y.lo;
		auto t5 = (uint128) x.mid * y.mid;
		auto t6 = x.mid * y.hi;
		auto t7 = x.hi * y.lo;
		auto t8 = x.hi * y.mid;

		auto lo = (uint64_t) t1;
		auto m1 = (t1 >> 64) + (uint64_t) t2;
		auto m2 = (uint64_t) m1;
		auto mid = (uint128) m2 + (uint64_t) t4;
		auto hi = (t2 >> 64) + t3 + (t4 >> 64) + t5 + (t6 << 64) + t7
			 + (t8 << 64) + (m1 >> 64) + (mid >> 64);

		return {lo, (uint64_t)mid, hi};
	}

	uint512 mul512(uint256 x, uint256 y)
	{
		auto x_lo = (uint128) x;
		auto y_lo = (uint128) y;

		auto t1 = mul(x_lo, y_lo);
		auto t2 = mul(x_lo, y.hi);
		auto t3 = mul(x.hi, y_lo);
		auto t4 = mul(x.hi, y.hi);

		auto lo = (uint128) t1;
		auto mid = (uint256) t1.hi + (uint128) t2 + (uint128) t3;
		auto hi = (uint256)t2.hi + t3.hi + t4 + mid.hi;

		return {lo, (uint128)mid, hi};
	}

	inline void mul(i256* x, i256* y)
	{
		auto a = (uint256*) x;
		auto b = (uint256*) y;
		*a = mul(*a, *b);
	}

	bool isZero(i256 const* _n)
	{
		return _n->a == 0 && _n->b == 0 && _n->c == 0 && _n->d == 0;
	}

	const auto nLimbs = sizeof(i256) / sizeof(mp_limb_t);

	int countLimbs(i256 const* _n)
	{
		static const auto limbsInWord = sizeof(_n->a) / sizeof(mp_limb_t);
		static_assert(limbsInWord == 1, "E?");

		int l = nLimbs;
		if (_n->d != 0) return l;
		l -= limbsInWord;
		if (_n->c != 0) return l;
		l -= limbsInWord;
		if (_n->b != 0) return l;
		l -= limbsInWord;
		if (_n->a != 0) return l;
		return 0;
	}
}

}
}
}


extern "C"
{

	using namespace dev::eth::jit;

	EXPORT void debug(uint64_t a, uint64_t b, uint64_t c, uint64_t d, char z)
	{
		std::cerr << "DEBUG " << z << ": " << d << c << b << a << std::endl;
	}

	EXPORT void arith_mul(uint256* _arg1, uint256* _arg2, uint256* o_result)
	{
		*o_result = mul(*_arg1, *_arg2);
	}

	EXPORT void arith_mul512(uint256* _arg1, uint256* _arg2, uint512* o_result)
	{
		*o_result = mul512(*_arg1, *_arg2);
	}

	EXPORT void arith_addmod(i256* _arg1, i256* _arg2, i256* _arg3, i256* o_result)
	{
		*o_result = {};
		if (isZero(_arg3))
			return;

		mpz_t x{nLimbs, countLimbs(_arg1), reinterpret_cast<mp_limb_t*>(_arg1)};
		mpz_t y{nLimbs, countLimbs(_arg2), reinterpret_cast<mp_limb_t*>(_arg2)};
		mpz_t m{nLimbs, countLimbs(_arg3), reinterpret_cast<mp_limb_t*>(_arg3)};
		mpz_t z{nLimbs, 0, reinterpret_cast<mp_limb_t*>(o_result)};
		static mp_limb_t s_limbs[nLimbs + 1] = {};
		static mpz_t s{nLimbs + 1, 0, &s_limbs[0]};

		mpz_add(s, x, y);
		mpz_tdiv_r(z, s, m);
	}

}


