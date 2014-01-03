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

u256 extractSender(u256 _v, u256 _r, u256 _s)
{
	// TODO...
	return _s;
}

template <class _T>
inline _T low160(_T const& _t)
{
	return _t & ((((_T)1) << 160) - 1);
}

bool State::transact(bytes const& _rlp)
{
	RLP rlp(_rlp);
	if (!rlp.isList())
		return false;
	RLPs items = rlp.toList();

//	if (!items[0].isFixedInt())
//		return false;
	if (!items[0].isString())
		return false;
	u256 nonce = items[0].toFatInt();

//	if (!(items[1].isEmpty() || items[1].isFixedInt()))
//		return false;
	if (!items[1].isString())
		return false;
	u256 address = items[1].toFatInt();

	if (!items[2].isFixedInt())
		return false;
	u256 value = items[2].toFatInt();

	if (!items[3].isFixedInt())
		return false;
	u256 fee = items[3].toFatInt();

	if (!items[4].isList())
		return false;
	u256s data;
	data.reserve(items[4].itemCount());
	for (auto const& i: items[4].toList())
		if (i.isFixedInt())
			data.push_back(i.toFatInt());
		else
			return false;

	if (!items[5].isString())
		return false;
	u256 v = items[5].toFatInt();

	if (!items[6].isString())
		return false;
	u256 r = items[6].toFatInt();

	if (!items[7].isString())
		return false;
	u256 s = items[7].toFatInt();

	u256 sender;
	try
	{
		sender = extractSender(v, r, s);
	}
	catch (...)
	{
		// Invalid signiture.
		// Error reporting?
		return false;
	}

	if (nonce != transactionsFrom(sender))
	{
		// Nonce is wrong.
		// Error reporting?
		return false;
	}

	if (balance(sender) < value + fee)
	{
		// Sender balance too low.
		// Error reporting?
		return false;
	}

	if (address)
	{
		assert(subBalance(sender, value));
		addBalance(address, value);

		if (isContractAddress(address))
		{
			bool ret = true;
			u256 minerFee = 0;

			try
			{
				execute(address, sender, value, fee, data, &minerFee);
			}
			catch (...)
			{
				// Execution error.
				// Error reporting?
				ret = false;
			}

			addBalance(m_minerAddress, minerFee);
			return ret;
		}
		else
			return true;
	}
	else
	{
		if (fee < data.size() * c_memoryFee + c_newContractFee)
		{
			// Fee too small.
			// Error reporting?
			return false;
		}

		u256 newAddress = low160(sha256(_rlp));
		if (isContractAddress(newAddress))
		{
			// Contract collision.
			// Error reporting?
			return false;
		}

		auto& mem = m_contractMemory[newAddress];
		for (uint i = 0; i < data.size(); ++i)
			mem[i] = data[i];
		assert(subBalance(sender, value));
		addBalance(newAddress, value);
		return true;
	}
}

void State::execute(u256 _myAddress, u256 _txSender, u256 _txValue, u256 _txFee, u256s const& _txData, u256* _totalFee)
{
	std::vector<u256> stack;
	auto& myMemory = ensureMemory(_myAddress);

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
			stack.push_back(m_currentBlock.coinbase);
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
				// NOTE: this aligns to right of 256-bit container (low-order bytes). This won't work if they're treated as byte-arrays and thus left-aligned in a 256-bit container.
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
			auto contractAddress = stack.back();
			stack.back() = memory(contractAddress, memoryAddress);
			break;
		}
		case Instruction::BALANCE:
		{
			require(1);
			stack.back() = balance(stack.back());
			break;
		}
		case Instruction::MKTX:
		{
			require(4);
			auto dest = stack.back();
			stack.pop_back();

			auto value = stack.back();
			stack.pop_back();

			auto fee = stack.back();
			stack.pop_back();

			auto itemCount = stack.back();
			stack.pop_back();
			if (stack.size() < itemCount)
				throw OperandOutOfRange(0, stack.size(), itemCount);
			u256s data;
			data.reserve((uint)itemCount);
			for (auto i = 0; i < itemCount; ++i)
			{
				data.push_back(stack.back());
				stack.pop_back();
			}

			u256 nonce = transactionsFrom(_myAddress);

			u256 v = 42;	// TODO: turn our address into a v/r/s signature?
			u256 r = 42;
			u256 s = _myAddress;
			// v/r/s are required to make the transaction hash (via the RLP serialisation) and thus are required in the creation of a contract.

			RLPStream rlp;
			if (nonce)
				rlp << nonce;
			else
				rlp << "";
			if (dest)
				rlp << toBigEndianString(dest);
			else
				rlp << "";
			rlp << value << fee << data << toBigEndianString(v) << toBigEndianString(r) << toBigEndianString(s);
			transact(rlp.out());

			break;
		}
		case Instruction::SUICIDE:
		{
			require(1);
			auto dest = stack.back();
			u256 minusVoidFee = m_contractMemory[_myAddress].size() * c_memoryFee;
			addBalance(dest, balance(_myAddress) + minusVoidFee - _txFee);
			m_balance.erase(_myAddress);
			m_contractMemory.erase(_myAddress);
			// ...follow through to...
		}
		case Instruction::STOP:
			return;
		default:
			throw BadInstruction();
		}
	}
}
