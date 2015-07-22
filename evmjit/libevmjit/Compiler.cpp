#include "Compiler.h"

#include <functional>
#include <fstream>
#include <chrono>
#include <sstream>

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/CFG.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IntrinsicInst.h>
#include "preprocessor/llvm_includes_end.h"

#include "Instruction.h"
#include "Type.h"
#include "Memory.h"
#include "Stack.h"
#include "Ext.h"
#include "GasMeter.h"
#include "Utils.h"
#include "Endianness.h"
#include "Arith256.h"
#include "RuntimeManager.h"

namespace dev
{
namespace eth
{
namespace jit
{

Compiler::Compiler(Options const& _options):
	m_options(_options),
	m_builder(llvm::getGlobalContext())
{
	Type::init(m_builder.getContext());
}

void Compiler::createBasicBlocks(code_iterator _codeBegin, code_iterator _codeEnd)
{
	/// Helper function that skips push data and finds next iterator (can be the end)
	auto skipPushDataAndGetNext = [](code_iterator _curr, code_iterator _end)
	{
		static const auto push1  = static_cast<size_t>(Instruction::PUSH1);
		static const auto push32 = static_cast<size_t>(Instruction::PUSH32);
		size_t offset = 1;
		if (*_curr >= push1 && *_curr <= push32)
			offset += std::min<size_t>(*_curr - push1 + 1, (_end - _curr) - 1);
		return _curr + offset;
	};

	// Skip all STOPs in the end
	for (; _codeEnd != _codeBegin; --_codeEnd)
		if (*(_codeEnd - 1) != static_cast<byte>(Instruction::STOP))
			break;

	auto begin = _codeBegin; // begin of current block
	bool nextJumpDest = false;
	for (auto curr = begin, next = begin; curr != _codeEnd; curr = next)
	{
		next = skipPushDataAndGetNext(curr, _codeEnd);

		bool isEnd = false;
		switch (Instruction(*curr))
		{
		case Instruction::JUMP:
		case Instruction::JUMPI:
		case Instruction::RETURN:
		case Instruction::STOP:
		case Instruction::SUICIDE:
			isEnd = true;
			break;

		case Instruction::JUMPDEST:
			nextJumpDest = true;
			break;

		default:
			break;
		}

		assert(next <= _codeEnd);
		if (next == _codeEnd || Instruction(*next) == Instruction::JUMPDEST)
			isEnd = true;

		if (isEnd)
		{
			auto beginIdx = begin - _codeBegin;
			m_basicBlocks.emplace(std::piecewise_construct, std::forward_as_tuple(beginIdx),
					std::forward_as_tuple(beginIdx, begin, next, m_mainFunc, m_builder, nextJumpDest));
			nextJumpDest = false;
			begin = next;
		}
	}
}

llvm::BasicBlock* Compiler::getJumpTableBlock(RuntimeManager& _runtimeManager)
{
	if (!m_jumpTableBlock)
	{
		m_jumpTableBlock.reset(new BasicBlock("JumpTable", m_mainFunc, m_builder, true));
		InsertPointGuard g{m_builder};
		m_builder.SetInsertPoint(m_jumpTableBlock->llvm());
		auto dest = m_builder.CreatePHI(Type::Word, 8, "target");
		auto switchInstr = m_builder.CreateSwitch(dest, getBadJumpBlock(_runtimeManager));
		for (auto&& p : m_basicBlocks)
		{
			if (p.second.isJumpDest())
				switchInstr->addCase(Constant::get(p.first), p.second.llvm());
		}
	}
	return m_jumpTableBlock->llvm();
}

llvm::BasicBlock* Compiler::getBadJumpBlock(RuntimeManager& _runtimeManager)
{
	if (!m_badJumpBlock)
	{
		m_badJumpBlock.reset(new BasicBlock("BadJump", m_mainFunc, m_builder, true));
		InsertPointGuard g{m_builder};
		m_builder.SetInsertPoint(m_badJumpBlock->llvm());
		_runtimeManager.exit(ReturnCode::BadJumpDestination);
	}
	return m_badJumpBlock->llvm();
}

std::unique_ptr<llvm::Module> Compiler::compile(code_iterator _begin, code_iterator _end, std::string const& _id)
{
	auto module = std::unique_ptr<llvm::Module>(new llvm::Module(_id, m_builder.getContext()));

	// Create main function
	auto mainFuncType = llvm::FunctionType::get(Type::MainReturn, Type::RuntimePtr, false);
	m_mainFunc = llvm::Function::Create(mainFuncType, llvm::Function::ExternalLinkage, _id, module.get());
	m_mainFunc->getArgumentList().front().setName("rt");

	// Create entry basic block
	auto entryBlock = llvm::BasicBlock::Create(m_builder.getContext(), {}, m_mainFunc);
	m_builder.SetInsertPoint(entryBlock);

	createBasicBlocks(_begin, _end);

	// Init runtime structures.
	RuntimeManager runtimeManager(m_builder, _begin, _end);
	GasMeter gasMeter(m_builder, runtimeManager);
	Memory memory(runtimeManager, gasMeter);
	Ext ext(runtimeManager, memory);
	Stack stack(m_builder, runtimeManager);
	runtimeManager.setStack(stack); // Runtime Manager will free stack memory
	Arith256 arith(m_builder);

	auto jmpBufWords = m_builder.CreateAlloca(Type::BytePtr, m_builder.getInt64(3), "jmpBuf.words");
	auto frameaddress = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::frameaddress);
	auto fp = m_builder.CreateCall(frameaddress, m_builder.getInt32(0), "fp");
	m_builder.CreateStore(fp, jmpBufWords);
	auto stacksave = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::stacksave);
	auto sp = m_builder.CreateCall(stacksave, {}, "sp");
	auto jmpBufSp = m_builder.CreateConstInBoundsGEP1_64(jmpBufWords, 2, "jmpBuf.sp");
	m_builder.CreateStore(sp, jmpBufSp);
	auto setjmp = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::eh_sjlj_setjmp);
	auto jmpBuf = m_builder.CreateBitCast(jmpBufWords, Type::BytePtr, "jmpBuf");
	auto r = m_builder.CreateCall(setjmp, jmpBuf);
	auto normalFlow = m_builder.CreateICmpEQ(r, m_builder.getInt32(0));
	runtimeManager.setJmpBuf(jmpBuf);

	// TODO: Create Stop basic block on demand
	m_stopBB = llvm::BasicBlock::Create(m_mainFunc->getContext(), "Stop", m_mainFunc);
	m_abortBB = llvm::BasicBlock::Create(m_mainFunc->getContext(), "Abort", m_mainFunc);

	auto firstBB = m_basicBlocks.empty() ? m_stopBB : m_basicBlocks.begin()->second.llvm();
	m_builder.CreateCondBr(normalFlow, firstBB, m_abortBB, Type::expectTrue);

	for (auto basicBlockPairIt = m_basicBlocks.begin(); basicBlockPairIt != m_basicBlocks.end(); ++basicBlockPairIt)
	{
		auto& basicBlock = basicBlockPairIt->second;
		auto iterCopy = basicBlockPairIt;
		++iterCopy;
		auto nextBasicBlock = (iterCopy != m_basicBlocks.end()) ? iterCopy->second.llvm() : nullptr;
		compileBasicBlock(basicBlock, runtimeManager, arith, memory, ext, gasMeter, nextBasicBlock);
	}

	// Code for special blocks:
	// TODO: move to separate function.
	m_builder.SetInsertPoint(m_stopBB);
	runtimeManager.exit(ReturnCode::Stop);

	m_builder.SetInsertPoint(m_abortBB);
	runtimeManager.exit(ReturnCode::OutOfGas);

	removeDeadBlocks();

	// Link jump table target index
	if (m_jumpTableBlock)
	{
		auto phi = llvm::cast<llvm::PHINode>(&m_jumpTableBlock->llvm()->getInstList().front());
		for (auto predIt = llvm::pred_begin(m_jumpTableBlock->llvm()); predIt != llvm::pred_end(m_jumpTableBlock->llvm()); ++predIt)
		{
			BasicBlock* pred = nullptr;
			for (auto&& p : m_basicBlocks)
			{
				if (p.second.llvm() == *predIt)
				{
					pred = &p.second;
					break;
				}
			}

			phi->addIncoming(pred->getJumpTarget(), pred->llvm());
		}
	}

	dumpCFGifRequired("blocks-init.dot");

	if (m_options.optimizeStack)
	{
		std::vector<BasicBlock*> blockList;
		for	(auto& entry : m_basicBlocks)
			blockList.push_back(&entry.second);

		if (m_jumpTableBlock)
			blockList.push_back(m_jumpTableBlock.get());

		BasicBlock::linkLocalStacks(blockList, m_builder);

		dumpCFGifRequired("blocks-opt.dot");
	}

	for (auto& entry : m_basicBlocks)
		entry.second.synchronizeLocalStack(stack);
	if (m_jumpTableBlock)
		m_jumpTableBlock->synchronizeLocalStack(stack);

	dumpCFGifRequired("blocks-sync.dot");

	return module;
}


void Compiler::compileBasicBlock(BasicBlock& _basicBlock, RuntimeManager& _runtimeManager,
								 Arith256& _arith, Memory& _memory, Ext& _ext, GasMeter& _gasMeter, llvm::BasicBlock* _nextBasicBlock)
{
	if (!_nextBasicBlock) // this is the last block in the code
		_nextBasicBlock = m_stopBB;

	m_builder.SetInsertPoint(_basicBlock.llvm());
	auto& stack = _basicBlock.localStack();

	for (auto it = _basicBlock.begin(); it != _basicBlock.end(); ++it)
	{
		auto inst = Instruction(*it);

		_gasMeter.count(inst);

		switch (inst)
		{

		case Instruction::ADD:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto result = m_builder.CreateAdd(lhs, rhs);
			stack.push(result);
			break;
		}

		case Instruction::SUB:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto result = m_builder.CreateSub(lhs, rhs);
			stack.push(result);
			break;
		}

		case Instruction::MUL:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res = m_builder.CreateMul(lhs, rhs);
			stack.push(res);
			break;
		}

		case Instruction::DIV:
		{
			auto d = stack.pop();
			auto n = stack.pop();
			auto divByZero = m_builder.CreateICmpEQ(n, Constant::get(0));
			n = m_builder.CreateSelect(divByZero, Constant::get(1), n); // protect against hardware signal
			auto r = m_builder.CreateUDiv(d, n);
			r = m_builder.CreateSelect(divByZero, Constant::get(0), r);
			stack.push(r);
			break;
		}

		case Instruction::SDIV:
		{
			auto d = stack.pop();
			auto n = stack.pop();
			auto divByZero = m_builder.CreateICmpEQ(n, Constant::get(0));
			auto divByMinusOne = m_builder.CreateICmpEQ(n, Constant::get(-1));
			n = m_builder.CreateSelect(divByZero, Constant::get(1), n); // protect against hardware signal
			auto r = m_builder.CreateSDiv(d, n);
			r = m_builder.CreateSelect(divByZero, Constant::get(0), r);
			auto dNeg = m_builder.CreateSub(Constant::get(0), d);
			r = m_builder.CreateSelect(divByMinusOne, dNeg, r); // protect against undef i256.min / -1
			stack.push(r);
			break;
		}

		case Instruction::MOD:
		{
			auto d = stack.pop();
			auto n = stack.pop();
			auto divByZero = m_builder.CreateICmpEQ(n, Constant::get(0));
			n = m_builder.CreateSelect(divByZero, Constant::get(1), n); // protect against hardware signal
			auto r = m_builder.CreateURem(d, n);
			r = m_builder.CreateSelect(divByZero, Constant::get(0), r);
			stack.push(r);
			break;
		}

		case Instruction::SMOD:
		{
			auto d = stack.pop();
			auto n = stack.pop();
			auto divByZero = m_builder.CreateICmpEQ(n, Constant::get(0));
			auto divByMinusOne = m_builder.CreateICmpEQ(n, Constant::get(-1));
			n = m_builder.CreateSelect(divByZero, Constant::get(1), n); // protect against hardware signal
			auto r = m_builder.CreateSRem(d, n);
			r = m_builder.CreateSelect(divByZero, Constant::get(0), r);
			r = m_builder.CreateSelect(divByMinusOne, Constant::get(0), r); // protect against undef i256.min / -1
			stack.push(r);
			break;
		}

		case Instruction::ADDMOD:
		{
			auto i512Ty = m_builder.getIntNTy(512);
			auto a = stack.pop();
			auto b = stack.pop();
			auto m = stack.pop();
			auto divByZero = m_builder.CreateICmpEQ(m, Constant::get(0));
			a = m_builder.CreateZExt(a, i512Ty);
			b = m_builder.CreateZExt(b, i512Ty);
			m = m_builder.CreateZExt(m, i512Ty);
			auto s = m_builder.CreateNUWAdd(a, b);
			s = m_builder.CreateURem(s, m);
			s = m_builder.CreateTrunc(s, Type::Word);
			s = m_builder.CreateSelect(divByZero, Constant::get(0), s);
			stack.push(s);
			break;
		}

		case Instruction::MULMOD:
		{
			auto i512Ty = m_builder.getIntNTy(512);
			auto a = stack.pop();
			auto b = stack.pop();
			auto m = stack.pop();
			auto divByZero = m_builder.CreateICmpEQ(m, Constant::get(0));
			m = m_builder.CreateZExt(m, i512Ty);
			// TODO: Add support for i256 x i256 -> i512 in LowerEVM pass
			llvm::Value* p = m_builder.CreateCall(Arith256::getMul512Func(*_basicBlock.llvm()->getParent()->getParent()), {a, b});
			p = m_builder.CreateURem(p, m);
			p = m_builder.CreateTrunc(p, Type::Word);
			p = m_builder.CreateSelect(divByZero, Constant::get(0), p);
			stack.push(p);
			break;
		}

		case Instruction::EXP:
		{
			auto base = stack.pop();
			auto exponent = stack.pop();
			_gasMeter.countExp(exponent);
			auto ret = _arith.exp(base, exponent);
			stack.push(ret);
			break;
		}

		case Instruction::NOT:
		{
			auto value = stack.pop();
			auto ret = m_builder.CreateXor(value, Constant::get(-1), "bnot");
			stack.push(ret);
			break;
		}

		case Instruction::LT:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res1 = m_builder.CreateICmpULT(lhs, rhs);
			auto res256 = m_builder.CreateZExt(res1, Type::Word);
			stack.push(res256);
			break;
		}

		case Instruction::GT:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res1 = m_builder.CreateICmpUGT(lhs, rhs);
			auto res256 = m_builder.CreateZExt(res1, Type::Word);
			stack.push(res256);
			break;
		}

		case Instruction::SLT:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res1 = m_builder.CreateICmpSLT(lhs, rhs);
			auto res256 = m_builder.CreateZExt(res1, Type::Word);
			stack.push(res256);
			break;
		}

		case Instruction::SGT:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res1 = m_builder.CreateICmpSGT(lhs, rhs);
			auto res256 = m_builder.CreateZExt(res1, Type::Word);
			stack.push(res256);
			break;
		}

		case Instruction::EQ:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res1 = m_builder.CreateICmpEQ(lhs, rhs);
			auto res256 = m_builder.CreateZExt(res1, Type::Word);
			stack.push(res256);
			break;
		}

		case Instruction::ISZERO:
		{
			auto top = stack.pop();
			auto iszero = m_builder.CreateICmpEQ(top, Constant::get(0), "iszero");
			auto result = m_builder.CreateZExt(iszero, Type::Word);
			stack.push(result);
			break;
		}

		case Instruction::AND:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res = m_builder.CreateAnd(lhs, rhs);
			stack.push(res);
			break;
		}

		case Instruction::OR:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res = m_builder.CreateOr(lhs, rhs);
			stack.push(res);
			break;
		}

		case Instruction::XOR:
		{
			auto lhs = stack.pop();
			auto rhs = stack.pop();
			auto res = m_builder.CreateXor(lhs, rhs);
			stack.push(res);
			break;
		}

		case Instruction::BYTE:
		{
			const auto idx = stack.pop();
			auto value = Endianness::toBE(m_builder, stack.pop());

			auto idxValid = m_builder.CreateICmpULT(idx, Constant::get(32), "idxValid");
			auto bytes = m_builder.CreateBitCast(value, llvm::VectorType::get(Type::Byte, 32), "bytes");
			// TODO: Workaround for LLVM bug. Using big value of index causes invalid memory access.
			auto safeIdx = m_builder.CreateTrunc(idx, m_builder.getIntNTy(5));
			// TODO: Workaround for LLVM bug. DAG Builder used sext on index instead of zext
			safeIdx = m_builder.CreateZExt(safeIdx, Type::Size);
			auto byte = m_builder.CreateExtractElement(bytes, safeIdx, "byte");
			value = m_builder.CreateZExt(byte, Type::Word);
			value = m_builder.CreateSelect(idxValid, value, Constant::get(0));
			stack.push(value);
			break;
		}

		case Instruction::SIGNEXTEND:
		{
			auto idx = stack.pop();
			auto word = stack.pop();

			auto k32_ = m_builder.CreateTrunc(idx, m_builder.getIntNTy(5), "k_32");
			auto k32 = m_builder.CreateZExt(k32_, Type::Size);
			auto k32x8 = m_builder.CreateMul(k32, m_builder.getInt64(8), "kx8");

			// test for word >> (k * 8 + 7)
			auto bitpos = m_builder.CreateAdd(k32x8, m_builder.getInt64(7), "bitpos");
			auto bitposEx = m_builder.CreateZExt(bitpos, Type::Word);
			auto bitval = m_builder.CreateLShr(word, bitposEx, "bitval");
			auto bittest = m_builder.CreateTrunc(bitval, Type::Bool, "bittest");

			auto mask_ = m_builder.CreateShl(Constant::get(1), bitposEx);
			auto mask = m_builder.CreateSub(mask_, Constant::get(1), "mask");

			auto negmask = m_builder.CreateXor(mask, llvm::ConstantInt::getAllOnesValue(Type::Word), "negmask");
			auto val1 = m_builder.CreateOr(word, negmask);
			auto val0 = m_builder.CreateAnd(word, mask);

			auto kInRange = m_builder.CreateICmpULE(idx, llvm::ConstantInt::get(Type::Word, 30));
			auto result = m_builder.CreateSelect(kInRange,
												 m_builder.CreateSelect(bittest, val1, val0),
												 word);
			stack.push(result);
			break;
		}

		case Instruction::SHA3:
		{
			auto inOff = stack.pop();
			auto inSize = stack.pop();
			_memory.require(inOff, inSize);
			_gasMeter.countSha3Data(inSize);
			auto hash = _ext.sha3(inOff, inSize);
			stack.push(hash);
			break;
		}

		case Instruction::POP:
		{
			stack.pop();
			break;
		}

		case Instruction::ANY_PUSH:
		{
			auto value = readPushData(it, _basicBlock.end());
			stack.push(Constant::get(value));
			break;
		}

		case Instruction::ANY_DUP:
		{
			auto index = static_cast<size_t>(inst) - static_cast<size_t>(Instruction::DUP1);
			stack.dup(index);
			break;
		}

		case Instruction::ANY_SWAP:
		{
			auto index = static_cast<size_t>(inst) - static_cast<size_t>(Instruction::SWAP1) + 1;
			stack.swap(index);
			break;
		}

		case Instruction::MLOAD:
		{
			auto addr = stack.pop();
			auto word = _memory.loadWord(addr);
			stack.push(word);
			break;
		}

		case Instruction::MSTORE:
		{
			auto addr = stack.pop();
			auto word = stack.pop();
			_memory.storeWord(addr, word);
			break;
		}

		case Instruction::MSTORE8:
		{
			auto addr = stack.pop();
			auto word = stack.pop();
			_memory.storeByte(addr, word);
			break;
		}

		case Instruction::MSIZE:
		{
			auto word = _memory.getSize();
			stack.push(word);
			break;
		}

		case Instruction::SLOAD:
		{
			auto index = stack.pop();
			auto value = _ext.sload(index);
			stack.push(value);
			break;
		}

		case Instruction::SSTORE:
		{
			auto index = stack.pop();
			auto value = stack.pop();
			_gasMeter.countSStore(_ext, index, value);
			_ext.sstore(index, value);
			break;
		}

		case Instruction::JUMP:
		case Instruction::JUMPI:
		{
			llvm::BasicBlock* targetBlock = nullptr;
			auto target = stack.pop();
			if (auto constant = llvm::dyn_cast<llvm::ConstantInt>(target))
			{
				auto&& c = constant->getValue();
				auto targetIdx = c.getActiveBits() <= 64 ? c.getZExtValue() : -1;
				auto it = m_basicBlocks.find(targetIdx);
				targetBlock = (it != m_basicBlocks.end() && it->second.isJumpDest()) ? it->second.llvm() : getBadJumpBlock(_runtimeManager);
			}

			// TODO: Improve; check for constants
			if (inst == Instruction::JUMP)
			{
				if (targetBlock)
				{
					m_builder.CreateBr(targetBlock);
				}
				else
				{
					_basicBlock.setJumpTarget(target);
					m_builder.CreateBr(getJumpTableBlock(_runtimeManager));
				}
			}
			else // JUMPI
			{
				auto val = stack.pop();
				auto zero = Constant::get(0);
				auto cond = m_builder.CreateICmpNE(val, zero, "nonzero");

				if (targetBlock)
				{
					m_builder.CreateCondBr(cond, targetBlock, _nextBasicBlock);
				}
				else
				{
					_basicBlock.setJumpTarget(target);
					m_builder.CreateCondBr(cond, getJumpTableBlock(_runtimeManager), _nextBasicBlock);
				}
			}
			break;
		}

		case Instruction::JUMPDEST:
		{
			// Nothing to do
			break;
		}

		case Instruction::PC:
		{
			auto value = Constant::get(it - _basicBlock.begin() + _basicBlock.firstInstrIdx());
			stack.push(value);
			break;
		}

		case Instruction::GAS:
		{
			_gasMeter.commitCostBlock();
			stack.push(m_builder.CreateZExt(_runtimeManager.getGas(), Type::Word));
			break;
		}

		case Instruction::ADDRESS:
		case Instruction::CALLER:
		case Instruction::ORIGIN:
		case Instruction::CALLVALUE:
		case Instruction::GASPRICE:
		case Instruction::COINBASE:
		case Instruction::DIFFICULTY:
		case Instruction::GASLIMIT:
		case Instruction::NUMBER:
		case Instruction::TIMESTAMP:
		{
			// Pushes an element of runtime data on stack
			auto value = _runtimeManager.get(inst);
			value = m_builder.CreateZExt(value, Type::Word);
			stack.push(value);
			break;
		}

		case Instruction::CODESIZE:
			stack.push(_runtimeManager.getCodeSize());
			break;

		case Instruction::CALLDATASIZE:
			stack.push(_runtimeManager.getCallDataSize());
			break;

		case Instruction::BLOCKHASH:
		{
			auto number = stack.pop();
			auto hash = _ext.blockHash(number);
			stack.push(hash);
			break;
		}

		case Instruction::BALANCE:
		{
			auto address = stack.pop();
			auto value = _ext.balance(address);
			stack.push(value);
			break;
		}

		case Instruction::EXTCODESIZE:
		{
			auto addr = stack.pop();
			auto codeRef = _ext.extcode(addr);
			stack.push(codeRef.size);
			break;
		}

		case Instruction::CALLDATACOPY:
		{
			auto destMemIdx = stack.pop();
			auto srcIdx = stack.pop();
			auto reqBytes = stack.pop();

			auto srcPtr = _runtimeManager.getCallData();
			auto srcSize = _runtimeManager.getCallDataSize();

			_memory.copyBytes(srcPtr, srcSize, srcIdx, destMemIdx, reqBytes);
			break;
		}

		case Instruction::CODECOPY:
		{
			auto destMemIdx = stack.pop();
			auto srcIdx = stack.pop();
			auto reqBytes = stack.pop();

			auto srcPtr = _runtimeManager.getCode();    // TODO: Code & its size are constants, feature #80814234
			auto srcSize = _runtimeManager.getCodeSize();

			_memory.copyBytes(srcPtr, srcSize, srcIdx, destMemIdx, reqBytes);
			break;
		}

		case Instruction::EXTCODECOPY:
		{
			auto addr = stack.pop();
			auto destMemIdx = stack.pop();
			auto srcIdx = stack.pop();
			auto reqBytes = stack.pop();

			auto codeRef = _ext.extcode(addr);

			_memory.copyBytes(codeRef.ptr, codeRef.size, srcIdx, destMemIdx, reqBytes);
			break;
		}

		case Instruction::CALLDATALOAD:
		{
			auto idx = stack.pop();
			auto value = _ext.calldataload(idx);
			stack.push(value);
			break;
		}

		case Instruction::CREATE:
		{
			auto endowment = stack.pop();
			auto initOff = stack.pop();
			auto initSize = stack.pop();
			_memory.require(initOff, initSize);

			_gasMeter.commitCostBlock();
			auto address = _ext.create(endowment, initOff, initSize);
			stack.push(address);
			break;
		}

		case Instruction::CALL:
		case Instruction::CALLCODE:
		{
			auto callGas = stack.pop();
			auto codeAddress = stack.pop();
			auto value = stack.pop();
			auto inOff = stack.pop();
			auto inSize = stack.pop();
			auto outOff = stack.pop();
			auto outSize = stack.pop();

			_gasMeter.commitCostBlock();

			// Require memory for in and out buffers
			_memory.require(outOff, outSize);	// Out buffer first as we guess it will be after the in one
			_memory.require(inOff, inSize);

			auto receiveAddress = codeAddress;
			if (inst == Instruction::CALLCODE)
				receiveAddress = _runtimeManager.get(RuntimeData::Address);

			auto ret = _ext.call(callGas, receiveAddress, value, inOff, inSize, outOff, outSize, codeAddress);
			_gasMeter.count(m_builder.getInt64(0), _runtimeManager.getJmpBuf(), _runtimeManager.getGasPtr());
			stack.push(ret);
			break;
		}

		case Instruction::RETURN:
		{
			auto index = stack.pop();
			auto size = stack.pop();

			_memory.require(index, size);
			_runtimeManager.registerReturnData(index, size);

			_runtimeManager.exit(ReturnCode::Return);
			break;
		}

		case Instruction::SUICIDE:
		{
			_runtimeManager.registerSuicide(stack.pop());
			_runtimeManager.exit(ReturnCode::Suicide);
			break;
		}


		case Instruction::STOP:
		{
			m_builder.CreateBr(m_stopBB);
			break;
		}

		case Instruction::LOG0:
		case Instruction::LOG1:
		case Instruction::LOG2:
		case Instruction::LOG3:
		case Instruction::LOG4:
		{
			auto beginIdx = stack.pop();
			auto numBytes = stack.pop();
			_memory.require(beginIdx, numBytes);

			// This will commit the current cost block
			_gasMeter.countLogData(numBytes);

			std::array<llvm::Value*, 4> topics{{}};
			auto numTopics = static_cast<size_t>(inst) - static_cast<size_t>(Instruction::LOG0);
			for (size_t i = 0; i < numTopics; ++i)
				topics[i] = stack.pop();

			_ext.log(beginIdx, numBytes, topics);
			break;
		}

		default: // Invalid instruction - abort
			m_builder.CreateBr(m_abortBB);
			it = _basicBlock.end() - 1; // finish block compilation
		}
	}

	_gasMeter.commitCostBlock();

	// Block may have no terminator if the next instruction is a jump destination.
	if (!_basicBlock.llvm()->getTerminator())
		m_builder.CreateBr(_nextBasicBlock);

	m_builder.SetInsertPoint(_basicBlock.llvm()->getFirstNonPHI());
	_runtimeManager.checkStackLimit(_basicBlock.localStack().getMaxSize(), _basicBlock.localStack().getDiff());
}



void Compiler::removeDeadBlocks()
{
	// Remove dead basic blocks
	auto sthErased = false;
	do
	{
		sthErased = false;
		for (auto it = m_basicBlocks.begin(); it != m_basicBlocks.end();)
		{
			auto llvmBB = it->second.llvm();
			if (llvm::pred_begin(llvmBB) == llvm::pred_end(llvmBB))
			{
				llvmBB->eraseFromParent();
				m_basicBlocks.erase(it++);
				sthErased = true;
			}
			else
				++it;
		}
	}
	while (sthErased);

	if (m_jumpTableBlock && llvm::pred_begin(m_jumpTableBlock->llvm()) == llvm::pred_end(m_jumpTableBlock->llvm()))
	{
		m_jumpTableBlock->llvm()->eraseFromParent();
		m_jumpTableBlock.reset();
	}

	if (m_badJumpBlock && llvm::pred_begin(m_badJumpBlock->llvm()) == llvm::pred_end(m_badJumpBlock->llvm()))
	{
		m_badJumpBlock->llvm()->eraseFromParent();
		m_badJumpBlock.reset();
	}
}

void Compiler::dumpCFGifRequired(std::string const& _dotfilePath)
{
	if (! m_options.dumpCFG)
		return;

	// TODO: handle i/o failures
	std::ofstream ofs(_dotfilePath);
	dumpCFGtoStream(ofs);
	ofs.close();
}

void Compiler::dumpCFGtoStream(std::ostream& _out)
{
	_out << "digraph BB {\n"
		 << "  node [shape=record, fontname=Courier, fontsize=10];\n"
		 << "  entry [share=record, label=\"entry block\"];\n";

	std::vector<BasicBlock*> blocks;
	for (auto& pair : m_basicBlocks)
		blocks.push_back(&pair.second);
	if (m_jumpTableBlock)
		blocks.push_back(m_jumpTableBlock.get());
	if (m_badJumpBlock)
		blocks.push_back(m_badJumpBlock.get());

	// std::map<BasicBlock*,int> phiNodesPerBlock;

	// Output nodes
	for (auto bb : blocks)
	{
		std::string blockName = bb->llvm()->getName();

		std::ostringstream oss;
		bb->dump(oss, true);

		_out << " \"" << blockName << "\" [shape=record, label=\" { " << blockName << "|" << oss.str() << "} \"];\n";
	}

	// Output edges
	for (auto bb : blocks)
	{
		std::string blockName = bb->llvm()->getName();

		auto end = llvm::pred_end(bb->llvm());
		for (llvm::pred_iterator it = llvm::pred_begin(bb->llvm()); it != end; ++it)
		{
			_out << "  \"" << (*it)->getName().str() << "\" -> \"" << blockName << "\" ["
				 << ((m_jumpTableBlock.get() && *it == m_jumpTableBlock.get()->llvm()) ? "style = dashed, " : "")
				 << "];\n";
		}
	}

	_out << "}\n";
}

void Compiler::dump()
{
	for (auto& entry : m_basicBlocks)
		entry.second.dump();
	if (m_jumpTableBlock != nullptr)
		m_jumpTableBlock->dump();
}

}
}
}
