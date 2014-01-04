#include <secp256k1.h>
#include <random>
#include "sha256.h"
#include "VirtualMachine.h"
using namespace std;
using namespace eth;

u256 const State::c_stepFee = 0;
u256 const State::c_dataFee = 0;
u256 const State::c_memoryFee = 0;
u256 const State::c_extroFee = 0;
u256 const State::c_cryptoFee = 0;
u256 const State::c_newContractFee = 0;

Transaction::Transaction(bytes const& _rlpData)
{
	RLP rlp(_rlpData);
	nonce = rlp[0].toFatIntFromString();
	receiveAddress = as160(rlp[1].toFatIntFromString());
	value = rlp[2].toFatIntStrict();
	fee = rlp[3].toFatIntStrict();
	data.reserve(rlp[4].itemCountStrict());
	for (auto const& i: rlp[4])
		data.push_back(i.toFatIntStrict());
	vrs = Signature{ rlp[5].toFatIntFromString(), rlp[6].toFatIntFromString(), rlp[7].toFatIntFromString() };
}

bytes Transaction::rlp() const
{
	RLPStream rlp;
	rlp << RLPList(8);
	if (nonce)
		rlp << nonce;
	else
		rlp << "";
	if (receiveAddress)
		rlp << toCompactBigEndianString(receiveAddress);
	else
		rlp << "";
	rlp << value << fee << data << toCompactBigEndianString(vrs.v) << toCompactBigEndianString(vrs.r) << toCompactBigEndianString(vrs.s);
	return rlp.out();
}

// Entry point for a user-originated transaction.
bool State::execute(Transaction const& _t)
{
	return execute(_t, _t.vrs.address());
}

bool State::execute(Transaction const& _t, u160 _sender)
{
	// Entry point for a contract-originated transaction.

	if (_t.nonce != transactionsFrom(_sender))
	{
		// Nonce is wrong.
		// Error reporting?
		return false;
	}

	if (balance(_sender) < _t.value + _t.fee)
	{
		// Sender balance too low.
		// Error reporting?
		return false;
	}

	if (_t.receiveAddress)
	{
		assert(subBalance(_sender, _t.value));
		addBalance(_t.receiveAddress, _t.value);

		if (isContractAddress(_t.receiveAddress))
		{
			u256 minerFee = 0;

			try
			{
				execute(_t.receiveAddress, _sender, _t.value, _t.fee, _t.data, &minerFee);
				addBalance(m_minerAddress, minerFee);
				return true;
			}
			catch (...)
			{
				// Execution error.
				// Error reporting?
				addBalance(m_minerAddress, minerFee);
				throw ExecutionException();
			}
		}
		else
			return true;
	}
	else
	{
		if (_t.fee < _t.data.size() * c_memoryFee + c_newContractFee)
		{
			// Fee too small.
			// Error reporting?
			return false;
		}

		u160 newAddress = low160(_t.sha256());

		if (isContractAddress(newAddress))
		{
			// Contract collision.
			// Error reporting?
			return false;
		}

		//
		auto& mem = m_current[newAddress].memory();
		for (uint i = 0; i < _t.data.size(); ++i)
			mem[i] = _t.data[i];
		assert(subBalance(_sender, _t.value));
		addBalance(newAddress, _t.value);
		return true;
	}
}

void State::execute(u160 _myAddress, u160 _txSender, u256 _txValue, u256 _txFee, u256s const& _txData, u256* _totalFee)
{
	std::vector<u256> stack;
	auto m = m_current.find(_myAddress);
	if (m == m_current.end())
		throw NoSuchContract();
	auto& myMemory = m->second.memory();

	auto require = [&](u256 _n)
	{
		if (stack.size() < _n)
			throw StackTooSmall(_n, stack.size());
	};
	auto mem = [&](u256 _n) -> u256
	{
		auto i = myMemory.find(_n);
		return i == myMemory.end() ? 0 : i->second;
	};
	auto setMem = [&](u256 _n, u256 _v)
	{
		if (_v)
			myMemory[_n] = _v;
		else
			myMemory.erase(_n);
	};

	u256 curPC = 0;
	u256 nextPC = 1;
	u256 stepCount = 0;
	for (bool stopped = false; !stopped; curPC = nextPC, nextPC = curPC + 1)
	{
		stepCount++;

		bigint minerFee = stepCount > 16 ? c_stepFee : 0;
		bigint voidFee = 0;

		auto rawInst = mem(curPC);
		if (rawInst > 0xff)
			throw BadInstruction();
		Instruction inst = (Instruction)(uint8_t)rawInst;

		switch (inst)
		{
		case Instruction::STORE:
			require(2);
			if (!mem(stack.back()) && stack[stack.size() - 2])
				voidFee += c_memoryFee;
			if (mem(stack.back()) && !stack[stack.size() - 2])
				voidFee -= c_memoryFee;
			// continue on to...
		case Instruction::LOAD:
			minerFee += c_dataFee;
			break;

		case Instruction::EXTRO:
		case Instruction::BALANCE:
			minerFee += c_extroFee;
			break;

		case Instruction::SHA256:
		case Instruction::RIPEMD160:
		case Instruction::ECMUL:
		case Instruction::ECADD:
		case Instruction::ECSIGN:
		case Instruction::ECRECOVER:
		case Instruction::ECVALID:
			minerFee += c_cryptoFee;
			break;
		default:
			break;
		}

		if (minerFee + voidFee > balance(_myAddress))
			return;
		subBalance(_myAddress, minerFee + voidFee);
		*_totalFee += (u256)minerFee;

		switch (inst)
		{
		case Instruction::ADD:
			//pops two items and pushes S[-1] + S[-2] mod 2^256.
			require(2);
			stack[stack.size() - 2] += stack.back();
			stack.pop_back();
			break;
		case Instruction::MUL:
			//pops two items and pushes S[-1] * S[-2] mod 2^256.
			require(2);
			stack[stack.size() - 2] *= stack.back();
			stack.pop_back();
			break;
		case Instruction::SUB:
			require(2);
			stack[stack.size() - 2] = stack.back() - stack[stack.size() - 2];
			stack.pop_back();
			break;
		case Instruction::DIV:
			require(2);
			stack[stack.size() - 2] = stack.back() / stack[stack.size() - 2];
			stack.pop_back();
			break;
		case Instruction::SDIV:
			require(2);
			(s256&)stack[stack.size() - 2] = (s256&)stack.back() / (s256&)stack[stack.size() - 2];
			stack.pop_back();
			break;
		case Instruction::MOD:
			require(2);
			stack[stack.size() - 2] = stack.back() % stack[stack.size() - 2];
			stack.pop_back();
			break;
		case Instruction::SMOD:
			require(2);
			(s256&)stack[stack.size() - 2] = (s256&)stack.back() % (s256&)stack[stack.size() - 2];
			stack.pop_back();
			break;
		case Instruction::EXP:
		{
			// TODO: better implementation?
			require(2);
			auto n = stack.back();
			auto x = stack[stack.size() - 2];
			stack.pop_back();
			for (u256 i = 0; i < x; ++i)
				n *= n;
			stack.back() = n;
			break;
		}
		case Instruction::NEG:
			require(1);
			stack.back() = ~(stack.back() - 1);
			break;
		case Instruction::LT:
			require(2);
			stack[stack.size() - 2] = stack.back() < stack[stack.size() - 2] ? 1 : 0;
			stack.pop_back();
			break;
		case Instruction::LE:
			require(2);
			stack[stack.size() - 2] = stack.back() <= stack[stack.size() - 2] ? 1 : 0;
			stack.pop_back();
			break;
		case Instruction::GT:
			require(2);
			stack[stack.size() - 2] = stack.back() > stack[stack.size() - 2] ? 1 : 0;
			stack.pop_back();
			break;
		case Instruction::GE:
			require(2);
			stack[stack.size() - 2] = stack.back() >= stack[stack.size() - 2] ? 1 : 0;
			stack.pop_back();
			break;
		case Instruction::EQ:
			require(2);
			stack[stack.size() - 2] = stack.back() == stack[stack.size() - 2] ? 1 : 0;
			stack.pop_back();
			break;
		case Instruction::NOT:
			require(1);
			stack.back() = stack.back() ? 0 : 1;
			stack.pop_back();
			break;
		case Instruction::MYADDRESS:
			stack.push_back(_myAddress);
			break;
		case Instruction::TXSENDER:
			stack.push_back(_txSender);
			break;
		case Instruction::TXVALUE:
			stack.push_back(_txValue);
			break;
		case Instruction::TXFEE:
			stack.push_back(_txFee);
			break;
		case Instruction::TXDATAN:
			stack.push_back(_txData.size());
			break;
		case Instruction::TXDATA:
			require(1);
			stack.back() = stack.back() < _txData.size() ? _txData[(uint)stack.back()] : 0;
			break;
		case Instruction::BLK_PREVHASH:
			stack.push_back(m_previousBlock.hash);
			break;
		case Instruction::BLK_COINBASE:
			stack.push_back(m_currentBlock.coinbaseAddress);
			break;
		case Instruction::BLK_TIMESTAMP:
			stack.push_back(m_currentBlock.timestamp);
			break;
		case Instruction::BLK_NUMBER:
			stack.push_back(m_currentBlock.number);
			break;
		case Instruction::BLK_DIFFICULTY:
			stack.push_back(m_currentBlock.difficulty);
			break;
		case Instruction::SHA256:
		case Instruction::RIPEMD160:
		{
			uint s = (uint)min(stack.back(), (u256)(stack.size() - 1) * 32);
			bytes b(s);
			uint i = 0;
			for (; s; s = (s >= 32 ? s - 32 : 0), i += 32)
			{
				stack.pop_back();
				u256 v = stack.back();
				int sz = (int)min<u256>(32, s) - 1;						// sz is one fewer than the number of bytes we're interested in.
				v >>= ((31 - sz) * 8);									// kill unused low-order bytes.
				for (int j = 0; j <= sz; ++j, v >>= 8)					// cycle through bytes, starting at low-order end.
					b[i + sz - j] = (byte)(v & 0xff);					// set each 32-byte (256-bit) chunk in reverse - (i.e. we want to put low-order last).
			}
			if (inst == Instruction::SHA256)
				stack.back() = sha256(b);
			else
				// NOTE: this aligns to right of 256-bit container (low-order bytes).
				// This won't work if they're treated as byte-arrays and thus left-aligned in a 256-bit container.
				stack.back() = ripemd160(&b);
			break;
		}
		case Instruction::ECMUL:
		case Instruction::ECADD:
		case Instruction::ECSIGN:
		case Instruction::ECRECOVER:
		case Instruction::ECVALID:
			// TODO
			break;
		case Instruction::PUSH:
		{
			stack.push_back(mem(curPC + 1));
			nextPC = curPC + 2;
			break;
		}
		case Instruction::POP:
			require(1);
			stack.pop_back();
			break;
		case Instruction::DUP:
			require(1);
			stack.push_back(stack.back());
			break;
		case Instruction::DUPN:
		{
			auto s = mem(curPC + 1);
			if (s == 0 || s > stack.size())
				throw OperandOutOfRange(1, stack.size(), s);
			stack.push_back(stack[stack.size() - (uint)s]);
			nextPC = curPC + 2;
			break;
		}
		case Instruction::SWAP:
		{
			require(2);
			auto d = stack.back();
			stack.back() = stack[stack.size() - 2];
			stack[stack.size() - 2] = d;
			break;
		}
		case Instruction::SWAPN:
		{
			require(1);
			auto d = stack.back();
			auto s = mem(curPC + 1);
			if (s == 0 || s > stack.size())
				throw OperandOutOfRange(1, stack.size(), s);
			stack.back() = stack[stack.size() - (uint)s];
			stack[stack.size() - (uint)s] = d;
			nextPC = curPC + 2;
			break;
		}
		case Instruction::LOAD:
			require(1);
			stack.back() = mem(stack.back());
			break;
		case Instruction::STORE:
			require(2);
			setMem(stack.back(), stack[stack.size() - 2]);
			stack.pop_back();
			stack.pop_back();
			break;
		case Instruction::JMP:
			require(1);
			nextPC = stack.back();
			stack.pop_back();
			break;
		case Instruction::JMPI:
			require(2);
			if (stack.back())
				nextPC = stack[stack.size() - 2];
			stack.pop_back();
			stack.pop_back();
			break;
		case Instruction::IND:
			stack.push_back(curPC);
			break;
		case Instruction::EXTRO:
		{
			require(2);
			auto memoryAddress = stack.back();
			stack.pop_back();
			u160 contractAddress = as160(stack.back());
			stack.back() = contractMemory(contractAddress, memoryAddress);
			break;
		}
		case Instruction::BALANCE:
		{
			require(1);
			stack.back() = balance(as160(stack.back()));
			break;
		}
		case Instruction::MKTX:
		{
			require(4);

			Transaction t;
			t.receiveAddress = as160(stack.back());
			stack.pop_back();
			t.value = stack.back();
			stack.pop_back();
			t.fee = stack.back();
			stack.pop_back();

			auto itemCount = stack.back();
			stack.pop_back();
			if (stack.size() < itemCount)
				throw OperandOutOfRange(0, stack.size(), itemCount);
			t.data.reserve((uint)itemCount);
			for (auto i = 0; i < itemCount; ++i)
			{
				t.data.push_back(stack.back());
				stack.pop_back();
			}

			t.nonce = transactionsFrom(_myAddress);
			execute(t);

			break;
		}
		case Instruction::SUICIDE:
		{
			require(1);
			u160 dest = as160(stack.back());
			u256 minusVoidFee = m_current[_myAddress].memory().size() * c_memoryFee;
			addBalance(dest, balance(_myAddress) + minusVoidFee);
			subBalance(dest, _txFee);
			m_current.erase(_myAddress);
			// ...follow through to...
		}
		case Instruction::STOP:
			return;
		default:
			throw BadInstruction();
		}
	}
}
