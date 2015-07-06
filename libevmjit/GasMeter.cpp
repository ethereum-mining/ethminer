#include "GasMeter.h"

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/IntrinsicInst.h>
#include "preprocessor/llvm_includes_end.h"

#include "Ext.h"
#include "RuntimeManager.h"

namespace dev
{
namespace eth
{
namespace jit
{

namespace // Helper functions
{

int64_t const c_stepGas[] = {0, 2, 3, 5, 8, 10, 20};
int64_t const c_expByteGas = 10;
int64_t const c_sha3Gas = 30;
int64_t const c_sha3WordGas = 6;
int64_t const c_sloadGas = 50;
int64_t const c_sstoreSetGas = 20000;
int64_t const c_sstoreResetGas = 5000;
int64_t const c_sstoreClearGas = 5000;
int64_t const c_jumpdestGas = 1;
int64_t const c_logGas = 375;
int64_t const c_logTopicGas = 375;
int64_t const c_logDataGas = 8;
int64_t const c_callGas = 40;
int64_t const c_createGas = 32000;
int64_t const c_memoryGas = 3;
int64_t const c_copyGas = 3;

int64_t getStepCost(Instruction inst)
{
	switch (inst)
	{
	// Tier 0
	case Instruction::STOP:
	case Instruction::RETURN:
	case Instruction::SUICIDE:
	case Instruction::SSTORE: // Handle cost of SSTORE separately in GasMeter::countSStore()
		return c_stepGas[0];

	// Tier 1
	case Instruction::ADDRESS:
	case Instruction::ORIGIN:
	case Instruction::CALLER:
	case Instruction::CALLVALUE:
	case Instruction::CALLDATASIZE:
	case Instruction::CODESIZE:
	case Instruction::GASPRICE:
	case Instruction::COINBASE:
	case Instruction::TIMESTAMP:
	case Instruction::NUMBER:
	case Instruction::DIFFICULTY:
	case Instruction::GASLIMIT:
	case Instruction::POP:
	case Instruction::PC:
	case Instruction::MSIZE:
	case Instruction::GAS:
		return c_stepGas[1];

	// Tier 2
	case Instruction::ADD:
	case Instruction::SUB:
	case Instruction::LT:
	case Instruction::GT:
	case Instruction::SLT:
	case Instruction::SGT:
	case Instruction::EQ:
	case Instruction::ISZERO:
	case Instruction::AND:
	case Instruction::OR:
	case Instruction::XOR:
	case Instruction::NOT:
	case Instruction::BYTE:
	case Instruction::CALLDATALOAD:
	case Instruction::CALLDATACOPY:
	case Instruction::CODECOPY:
	case Instruction::MLOAD:
	case Instruction::MSTORE:
	case Instruction::MSTORE8:
	case Instruction::ANY_PUSH:
	case Instruction::ANY_DUP:
	case Instruction::ANY_SWAP:
		return c_stepGas[2];

	// Tier 3
	case Instruction::MUL:
	case Instruction::DIV:
	case Instruction::SDIV:
	case Instruction::MOD:
	case Instruction::SMOD:
	case Instruction::SIGNEXTEND:
		return c_stepGas[3];

	// Tier 4
	case Instruction::ADDMOD:
	case Instruction::MULMOD:
	case Instruction::JUMP:
		return c_stepGas[4];

	// Tier 5
	case Instruction::EXP:
	case Instruction::JUMPI:
		return c_stepGas[5];

	// Tier 6
	case Instruction::BALANCE:
	case Instruction::EXTCODESIZE:
	case Instruction::EXTCODECOPY:
	case Instruction::BLOCKHASH:
		return c_stepGas[6];

	case Instruction::SHA3:
		return c_sha3Gas;

	case Instruction::SLOAD:
		return c_sloadGas;

	case Instruction::JUMPDEST:
		return c_jumpdestGas;

	case Instruction::LOG0:
	case Instruction::LOG1:
	case Instruction::LOG2:
	case Instruction::LOG3:
	case Instruction::LOG4:
	{
		auto numTopics = static_cast<int64_t>(inst) - static_cast<int64_t>(Instruction::LOG0);
		return c_logGas + numTopics * c_logTopicGas;
	}

	case Instruction::CALL:
	case Instruction::CALLCODE:
		return c_callGas;

	case Instruction::CREATE:
		return c_createGas;
	}

	return 0; // TODO: Add UNREACHABLE macro
}

}

GasMeter::GasMeter(llvm::IRBuilder<>& _builder, RuntimeManager& _runtimeManager) :
	CompilerHelper(_builder),
	m_runtimeManager(_runtimeManager)
{
	llvm::Type* gasCheckArgs[] = {Type::Gas->getPointerTo(), Type::Gas, Type::BytePtr};
	m_gasCheckFunc = llvm::Function::Create(llvm::FunctionType::get(Type::Void, gasCheckArgs, false), llvm::Function::PrivateLinkage, "gas.check", getModule());
	m_gasCheckFunc->setDoesNotThrow();
	m_gasCheckFunc->setDoesNotCapture(1);

	auto checkBB = llvm::BasicBlock::Create(_builder.getContext(), "Check", m_gasCheckFunc);
	auto updateBB = llvm::BasicBlock::Create(_builder.getContext(), "Update", m_gasCheckFunc);
	auto outOfGasBB = llvm::BasicBlock::Create(_builder.getContext(), "OutOfGas", m_gasCheckFunc);

	auto gasPtr = &m_gasCheckFunc->getArgumentList().front();
	gasPtr->setName("gasPtr");
	auto cost = gasPtr->getNextNode();
	cost->setName("cost");
	auto jmpBuf = cost->getNextNode();
	jmpBuf->setName("jmpBuf");

	InsertPointGuard guard(m_builder);
	m_builder.SetInsertPoint(checkBB);
	auto gas = m_builder.CreateLoad(gasPtr, "gas");
	auto gasUpdated = m_builder.CreateNSWSub(gas, cost, "gasUpdated");
	auto gasOk = m_builder.CreateICmpSGE(gasUpdated, m_builder.getInt64(0), "gasOk"); // gas >= 0, with gas == 0 we can still do 0 cost instructions
	m_builder.CreateCondBr(gasOk, updateBB, outOfGasBB, Type::expectTrue);

	m_builder.SetInsertPoint(updateBB);
	m_builder.CreateStore(gasUpdated, gasPtr);
	m_builder.CreateRetVoid();

	m_builder.SetInsertPoint(outOfGasBB);
	m_runtimeManager.abort(jmpBuf);
	m_builder.CreateUnreachable();
}

void GasMeter::count(Instruction _inst)
{
	if (!m_checkCall)
	{
		// Create gas check call with mocked block cost at begining of current cost-block
		m_checkCall = createCall(m_gasCheckFunc, {m_runtimeManager.getGasPtr(), llvm::UndefValue::get(Type::Gas), m_runtimeManager.getJmpBuf()});
	}

	m_blockCost += getStepCost(_inst);
}

void GasMeter::count(llvm::Value* _cost, llvm::Value* _jmpBuf, llvm::Value* _gasPtr)
{
	if (_cost->getType() == Type::Word)
	{
		auto gasMax256 = m_builder.CreateZExt(Constant::gasMax, Type::Word);
		auto tooHigh = m_builder.CreateICmpUGT(_cost, gasMax256, "costTooHigh");
		auto cost64 = m_builder.CreateTrunc(_cost, Type::Gas);
		_cost = m_builder.CreateSelect(tooHigh, Constant::gasMax, cost64, "cost");
	}

	assert(_cost->getType() == Type::Gas);
	createCall(m_gasCheckFunc, {_gasPtr ? _gasPtr : m_runtimeManager.getGasPtr(), _cost, _jmpBuf ? _jmpBuf : m_runtimeManager.getJmpBuf()});
}

void GasMeter::countExp(llvm::Value* _exponent)
{
	// Additional cost is 1 per significant byte of exponent
	// lz - leading zeros
	// cost = ((256 - lz) + 7) / 8

	// OPT: Can gas update be done in exp algorithm?
	auto ctlz = llvm::Intrinsic::getDeclaration(getModule(), llvm::Intrinsic::ctlz, Type::Word);
	auto lz256 = m_builder.CreateCall(ctlz, {_exponent, m_builder.getInt1(false)});
	auto lz = m_builder.CreateTrunc(lz256, Type::Gas, "lz");
	auto sigBits = m_builder.CreateSub(m_builder.getInt64(256), lz, "sigBits");
	auto sigBytes = m_builder.CreateUDiv(m_builder.CreateAdd(sigBits, m_builder.getInt64(7)), m_builder.getInt64(8));
	count(m_builder.CreateNUWMul(sigBytes, m_builder.getInt64(c_expByteGas)));
}

void GasMeter::countSStore(Ext& _ext, llvm::Value* _index, llvm::Value* _newValue)
{
	auto oldValue = _ext.sload(_index);
	auto oldValueIsZero = m_builder.CreateICmpEQ(oldValue, Constant::get(0), "oldValueIsZero");
	auto newValueIsntZero = m_builder.CreateICmpNE(_newValue, Constant::get(0), "newValueIsntZero");
	auto isInsert = m_builder.CreateAnd(oldValueIsZero, newValueIsntZero, "isInsert");
	static_assert(c_sstoreResetGas == c_sstoreClearGas, "Update SSTORE gas cost");
	auto cost = m_builder.CreateSelect(isInsert, m_builder.getInt64(c_sstoreSetGas), m_builder.getInt64(c_sstoreResetGas), "cost");
	count(cost);
}

void GasMeter::countLogData(llvm::Value* _dataLength)
{
	assert(m_checkCall);
	assert(m_blockCost > 0); // LOGn instruction is already counted
	static_assert(c_logDataGas != 1, "Log data gas cost has changed. Update GasMeter.");
	count(m_builder.CreateNUWMul(_dataLength, Constant::get(c_logDataGas))); // TODO: Use i64
}

void GasMeter::countSha3Data(llvm::Value* _dataLength)
{
	assert(m_checkCall);
	assert(m_blockCost > 0); // SHA3 instruction is already counted

	// TODO: This round ups to 32 happens in many places
	static_assert(c_sha3WordGas != 1, "SHA3 data cost has changed. Update GasMeter");
	auto dataLength64 = getBuilder().CreateTrunc(_dataLength, Type::Gas);
	auto words64 = m_builder.CreateUDiv(m_builder.CreateNUWAdd(dataLength64, getBuilder().getInt64(31)), getBuilder().getInt64(32));
	auto cost64 = m_builder.CreateNUWMul(getBuilder().getInt64(c_sha3WordGas), words64);
	count(cost64);
}

void GasMeter::giveBack(llvm::Value* _gas)
{
	assert(_gas->getType() == Type::Gas);
	m_runtimeManager.setGas(m_builder.CreateAdd(m_runtimeManager.getGas(), _gas));
}

void GasMeter::commitCostBlock()
{
	// If any uncommited block
	if (m_checkCall)
	{
		if (m_blockCost == 0) // Do not check 0
		{
			m_checkCall->eraseFromParent(); // Remove the gas check call
			m_checkCall = nullptr;
			return;
		}

		m_checkCall->setArgOperand(1, m_builder.getInt64(m_blockCost)); // Update block cost in gas check call
		m_checkCall = nullptr; // End cost-block
		m_blockCost = 0;
	}
	assert(m_blockCost == 0);
}

void GasMeter::countMemory(llvm::Value* _additionalMemoryInWords, llvm::Value* _jmpBuf, llvm::Value* _gasPtr)
{
	static_assert(c_memoryGas != 1, "Memory gas cost has changed. Update GasMeter.");
	count(_additionalMemoryInWords, _jmpBuf, _gasPtr);
}

void GasMeter::countCopy(llvm::Value* _copyWords)
{
	static_assert(c_copyGas != 1, "Copy gas cost has changed. Update GasMeter.");
	count(m_builder.CreateNUWMul(_copyWords, m_builder.getInt64(c_copyGas)));
}

}
}
}

