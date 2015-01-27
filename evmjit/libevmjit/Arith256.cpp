#include "Arith256.h"
#include "Runtime.h"
#include "Type.h"
#include "Endianness.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IntrinsicInst.h>
#include <gmp.h>

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
	m_div = Function::Create(FunctionType::get(Type::Void, arg2Types, false), Linkage::ExternalLinkage, "arith_div", getModule());
	m_mod = Function::Create(FunctionType::get(Type::Void, arg2Types, false), Linkage::ExternalLinkage, "arith_mod", getModule());
	m_sdiv = Function::Create(FunctionType::get(Type::Void, arg2Types, false), Linkage::ExternalLinkage, "arith_sdiv", getModule());
	m_smod = Function::Create(FunctionType::get(Type::Void, arg2Types, false), Linkage::ExternalLinkage, "arith_smod", getModule());
	m_exp = Function::Create(FunctionType::get(Type::Void, arg2Types, false), Linkage::ExternalLinkage, "arith_exp", getModule());
	m_addmod = Function::Create(FunctionType::get(Type::Void, arg3Types, false), Linkage::ExternalLinkage, "arith_addmod", getModule());
	m_mulmod = Function::Create(FunctionType::get(Type::Void, arg3Types, false), Linkage::ExternalLinkage, "arith_mulmod", getModule());
}

llvm::Function* Arith256::getDivFunc()
{
	if (!m_newDiv)
	{
		// Based of "Improved shift divisor algorithm" from "Software Integer Division" by Microsoft Research
		// The following algorithm also handles divisor of value 0 returning 0 for both quotient and reminder

		llvm::Type* argTypes[] = {Type::Word, Type::Word};
		auto retType = llvm::StructType::get(m_builder.getContext(), llvm::ArrayRef<llvm::Type*>{argTypes});
		m_newDiv = llvm::Function::Create(llvm::FunctionType::get(retType, argTypes, false), llvm::Function::PrivateLinkage, "arith.div", getModule());

		auto x = &m_newDiv->getArgumentList().front();
		x->setName("x");
		auto yArg = x->getNextNode();
		yArg->setName("y");

		InsertPointGuard guard{m_builder};

		auto entryBB = llvm::BasicBlock::Create(m_builder.getContext(), "Entry", m_newDiv);
		auto mainBB = llvm::BasicBlock::Create(m_builder.getContext(), "Main", m_newDiv);
		auto loopBB = llvm::BasicBlock::Create(m_builder.getContext(), "Loop", m_newDiv);
		auto continueBB = llvm::BasicBlock::Create(m_builder.getContext(), "Continue", m_newDiv);
		auto returnBB = llvm::BasicBlock::Create(m_builder.getContext(), "Return", m_newDiv);

		m_builder.SetInsertPoint(entryBB);
		auto yNonZero = m_builder.CreateICmpNE(yArg, Constant::get(0));
		auto yLEx = m_builder.CreateICmpULE(yArg, x);
		auto r0 = m_builder.CreateSelect(yNonZero, x, Constant::get(0), "r0");
		m_builder.CreateCondBr(m_builder.CreateAnd(yLEx, yNonZero), mainBB, returnBB);

		m_builder.SetInsertPoint(mainBB);
		auto ctlzIntr = llvm::Intrinsic::getDeclaration(getModule(), llvm::Intrinsic::ctlz, Type::Word);
		// both y and r are non-zero
		auto yLz = m_builder.CreateCall2(ctlzIntr, yArg, m_builder.getInt1(true), "y.lz");
		auto rLz = m_builder.CreateCall2(ctlzIntr, r0, m_builder.getInt1(true), "r.lz");
		auto i0 = m_builder.CreateNUWSub(yLz, rLz, "i0");
		auto shlBy0 = m_builder.CreateICmpEQ(i0, Constant::get(0));
		auto y0 = m_builder.CreateShl(yArg, i0);
		y0 = m_builder.CreateSelect(shlBy0, yArg, y0, "y0"); // Workaround for LLVM bug: shl by 0 produces wrong result
		m_builder.CreateBr(loopBB);

		m_builder.SetInsertPoint(loopBB);
		auto yPhi = m_builder.CreatePHI(Type::Word, 2, "y.phi");
		auto rPhi = m_builder.CreatePHI(Type::Word, 2, "r.phi");
		auto iPhi = m_builder.CreatePHI(Type::Word, 2, "i.phi");
		auto qPhi = m_builder.CreatePHI(Type::Word, 2, "q.phi");
		auto rUpdate = m_builder.CreateNUWSub(rPhi, yPhi);
		auto qUpdate = m_builder.CreateOr(qPhi, Constant::get(1));	// q += 1, q lowest bit is 0
		auto rGEy = m_builder.CreateICmpUGE(rPhi, yPhi);
		auto r1 = m_builder.CreateSelect(rGEy, rUpdate, rPhi, "r1");
		auto q1 = m_builder.CreateSelect(rGEy, qUpdate, qPhi, "q");
		auto iZero = m_builder.CreateICmpEQ(iPhi, Constant::get(0));
		m_builder.CreateCondBr(iZero, returnBB, continueBB);

		m_builder.SetInsertPoint(continueBB);
		auto i2 = m_builder.CreateNUWSub(iPhi, Constant::get(1));
		auto q2 = m_builder.CreateShl(q1, Constant::get(1));
		auto y2 = m_builder.CreateUDiv(yPhi, Constant::get(2));
		m_builder.CreateBr(loopBB);

		yPhi->addIncoming(y0, mainBB);
		yPhi->addIncoming(y2, continueBB);
		rPhi->addIncoming(r0, mainBB);
		rPhi->addIncoming(r1, continueBB);
		iPhi->addIncoming(i0, mainBB);
		iPhi->addIncoming(i2, continueBB);
		qPhi->addIncoming(Constant::get(0), mainBB);
		qPhi->addIncoming(q2, continueBB);

		m_builder.SetInsertPoint(returnBB);
		auto qRet = m_builder.CreatePHI(Type::Word, 2, "q.ret");
		qRet->addIncoming(Constant::get(0), entryBB);
		qRet->addIncoming(q1, loopBB);
		auto rRet = m_builder.CreatePHI(Type::Word, 2, "r.ret");
		rRet->addIncoming(r0, entryBB);
		rRet->addIncoming(r1, loopBB);
		auto ret = m_builder.CreateInsertValue(llvm::UndefValue::get(retType), qRet, 0, "ret0");
		ret = m_builder.CreateInsertValue(ret, rRet, 1, "ret");
		m_builder.CreateRet(ret);
	}
	return m_newDiv;
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

llvm::Value* Arith256::div(llvm::Value* _arg1, llvm::Value* _arg2)
{
	return m_builder.CreateExtractValue(createCall(getDivFunc(), {_arg1, _arg2}), 0, "div");
}

llvm::Value* Arith256::mod(llvm::Value* _arg1, llvm::Value* _arg2)
{
	return m_builder.CreateExtractValue(createCall(getDivFunc(), {_arg1, _arg2}), 1, "mod");
}

llvm::Value* Arith256::sdiv(llvm::Value* _arg1, llvm::Value* _arg2)
{
	return binaryOp(m_sdiv, _arg1, _arg2);
}

llvm::Value* Arith256::smod(llvm::Value* _arg1, llvm::Value* _arg2)
{
	return binaryOp(m_smod, _arg1, _arg2);
}

llvm::Value* Arith256::exp(llvm::Value* _arg1, llvm::Value* _arg2)
{
	return binaryOp(m_exp, _arg1, _arg2);
}

llvm::Value* Arith256::addmod(llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3)
{
	return ternaryOp(m_addmod, _arg1, _arg2, _arg3);
}

llvm::Value* Arith256::mulmod(llvm::Value* _arg1, llvm::Value* _arg2, llvm::Value* _arg3)
{
	return ternaryOp(m_mulmod, _arg1, _arg2, _arg3);
}

namespace
{
	using uint128 = __uint128_t;

//	uint128 add(uint128 a, uint128 b) { return a + b; }
//	uint128 mul(uint128 a, uint128 b) { return a * b; }
//
//	uint128 mulq(uint64_t x, uint64_t y)
//	{
//		return (uint128)x * (uint128)y;
//	}
//
//	uint128 addc(uint64_t x, uint64_t y)
//	{
//		return (uint128)x * (uint128)y;
//	}

	struct uint256
	{
		uint64_t lo;
		uint64_t mid;
		uint128 hi;
	};

//	uint256 add(uint256 x, uint256 y)
//	{
//		auto lo = (uint128) x.lo + y.lo;
//		auto mid = (uint128) x.mid + y.mid + (lo >> 64);
//		return {lo, mid, x.hi + y.hi + (mid >> 64)};
//	}

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

	bool isZero(i256 const* _n)
	{
		return _n->a == 0 && _n->b == 0 && _n->c == 0 && _n->d == 0;
	}

	const auto nLimbs = sizeof(i256) / sizeof(mp_limb_t);

	// FIXME: Not thread-safe
	static mp_limb_t mod_limbs[] = {0, 0, 0, 0, 1};
	static_assert(sizeof(mod_limbs) / sizeof(mod_limbs[0]) == nLimbs + 1, "mp_limb_t size mismatch");
	static const mpz_t mod{nLimbs + 1, nLimbs + 1, &mod_limbs[0]};

	static mp_limb_t tmp_limbs[nLimbs + 2];
	static mpz_t tmp{nLimbs + 2, 0, &tmp_limbs[0]};

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

	void u2s(mpz_t _u)
	{
		if (static_cast<std::make_signed<mp_limb_t>::type>(_u->_mp_d[nLimbs - 1]) < 0)
		{
			mpz_sub(tmp, mod, _u);
			mpz_set(_u, tmp);
			_u->_mp_size = -_u->_mp_size;
		}
	}

	void s2u(mpz_t _s)
	{
		if (_s->_mp_size < 0)
		{
			mpz_add(tmp, mod, _s);
			mpz_set(_s, tmp);
		}
	}
}

}
}
}


extern "C"
{

	using namespace dev::eth::jit;

	EXPORT void arith_mul(uint256* _arg1, uint256* _arg2, uint256* o_result)
	{
		*o_result = mul(*_arg1, *_arg2);
	}

	EXPORT void arith_sdiv(i256* _arg1, i256* _arg2, i256* o_result)
	{
		*o_result = {};
		if (isZero(_arg2))
			return;

		mpz_t x{nLimbs, countLimbs(_arg1), reinterpret_cast<mp_limb_t*>(_arg1)};
		mpz_t y{nLimbs, countLimbs(_arg2), reinterpret_cast<mp_limb_t*>(_arg2)};
		mpz_t z{nLimbs, 0, reinterpret_cast<mp_limb_t*>(o_result)};
		u2s(x);
		u2s(y);
		mpz_tdiv_q(z, x, y);
		s2u(z);
	}

	EXPORT void arith_smod(i256* _arg1, i256* _arg2, i256* o_result)
	{
		*o_result = {};
		if (isZero(_arg2))
			return;

		mpz_t x{nLimbs, countLimbs(_arg1), reinterpret_cast<mp_limb_t*>(_arg1)};
		mpz_t y{nLimbs, countLimbs(_arg2), reinterpret_cast<mp_limb_t*>(_arg2)};
		mpz_t z{nLimbs, 0, reinterpret_cast<mp_limb_t*>(o_result)};
		u2s(x);
		u2s(y);
		mpz_tdiv_r(z, x, y);
		s2u(z);
	}

	EXPORT void arith_exp(i256* _arg1, i256* _arg2, i256* o_result)
	{
		*o_result = {};

		static mp_limb_t mod_limbs[nLimbs + 1] = {};
		mod_limbs[nLimbs] = 1;
		static const mpz_t mod{nLimbs + 1, nLimbs + 1, &mod_limbs[0]};

		mpz_t x{nLimbs, countLimbs(_arg1), reinterpret_cast<mp_limb_t*>(_arg1)};
		mpz_t y{nLimbs, countLimbs(_arg2), reinterpret_cast<mp_limb_t*>(_arg2)};
		mpz_t z{nLimbs, 0, reinterpret_cast<mp_limb_t*>(o_result)};

		mpz_powm(z, x, y, mod);
	}

	EXPORT void arith_mulmod(i256* _arg1, i256* _arg2, i256* _arg3, i256* o_result)
	{
		*o_result = {};
		if (isZero(_arg3))
			return;

		mpz_t x{nLimbs, countLimbs(_arg1), reinterpret_cast<mp_limb_t*>(_arg1)};
		mpz_t y{nLimbs, countLimbs(_arg2), reinterpret_cast<mp_limb_t*>(_arg2)};
		mpz_t m{nLimbs, countLimbs(_arg3), reinterpret_cast<mp_limb_t*>(_arg3)};
		mpz_t z{nLimbs, 0, reinterpret_cast<mp_limb_t*>(o_result)};
		static mp_limb_t p_limbs[nLimbs * 2] = {};
		static mpz_t p{nLimbs * 2, 0, &p_limbs[0]};

		mpz_mul(p, x, y);
		mpz_tdiv_r(z, p, m);
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


