#include "sha256.h"
#include <secp256k1.h>
#include "VirtualMachine.h"
using namespace std;
using namespace eth;

VirtualMachine::~VirtualMachine()
{
}


void VirtualMachine::go()
{
	u256 curPC = 0;
	u256 nextPC = 1;
	auto& memory = m_state->memory(m_myAddress);

	auto require = [&](u256 _n)
	{
		if (m_stack.size() < _n)
			throw StackTooSmall(_n, m_stack.size());
	};
	auto mem = [&](u256 _n)
	{
		auto i = memory.find(_n);
		return i == memory.end() ? 0 : i->second;
	};

	for (bool stopped = false; !stopped; curPC = nextPC, nextPC = curPC + 1)
	{
		auto rawInst = mem(curPC);
		if (rawInst > 0xff)
			throw BadInstruction();
		Instruction inst = (Instruction)(uint8_t)rawInst;

		switch (inst)
		{
		case Instruction::ADD:
			//pops two items and pushes S[-1] + S[-2] mod 2^256.
			require(2);
			m_stack[m_stack.size() - 2] += m_stack.back();
			m_stack.pop_back();
			break;
		case Instruction::MUL:
			//pops two items and pushes S[-1] * S[-2] mod 2^256.
			require(2);
			m_stack[m_stack.size() - 2] *= m_stack.back();
			m_stack.pop_back();
			break;
		case Instruction::SUB:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() - m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			break;
		case Instruction::DIV:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() / m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			break;
		case Instruction::SDIV:
			require(2);
			(s256&)m_stack[m_stack.size() - 2] = (s256&)m_stack.back() / (s256&)m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			break;
		case Instruction::MOD:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() % m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			break;
		case Instruction::SMOD:
			require(2);
			(s256&)m_stack[m_stack.size() - 2] = (s256&)m_stack.back() % (s256&)m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			break;
		case Instruction::EXP:
		{
			// TODO: better implementation?
			require(2);
			auto n = m_stack.back();
			auto x = m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			for (u256 i = 0; i < x; ++i)
				n *= n;
			m_stack.back() = n;
			break;
		}
		case Instruction::NEG:
			require(1);
			m_stack.back() = ~(m_stack.back() - 1);
			break;
		case Instruction::LT:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() < m_stack[m_stack.size() - 2] ? 1 : 0;
			m_stack.pop_back();
			break;
		case Instruction::LE:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() <= m_stack[m_stack.size() - 2] ? 1 : 0;
			m_stack.pop_back();
			break;
		case Instruction::GT:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() > m_stack[m_stack.size() - 2] ? 1 : 0;
			m_stack.pop_back();
			break;
		case Instruction::GE:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() >= m_stack[m_stack.size() - 2] ? 1 : 0;
			m_stack.pop_back();
			break;
		case Instruction::EQ:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() == m_stack[m_stack.size() - 2] ? 1 : 0;
			m_stack.pop_back();
			break;
		case Instruction::NOT:
			require(1);
			m_stack.back() = m_stack.back() ? 0 : 1;
			m_stack.pop_back();
			break;
		case Instruction::MYADDRESS:
			m_stack.push_back(m_myAddress);
			break;
		case Instruction::TXSENDER:
			m_stack.push_back(m_txSender);
			break;
		case Instruction::TXVALUE:
			m_stack.push_back(m_txValue);
			break;
		case Instruction::TXFEE:
			m_stack.push_back(m_txFee);
			break;
		case Instruction::TXDATAN:
			m_stack.push_back(m_txData.size());
			break;
		case Instruction::TXDATA:
			require(1);
			m_stack.back() = m_stack.back() < m_txData.size() ? m_txData[(uint)m_stack.back()] : 0;
			break;
		case Instruction::BLK_PREVHASH:
			m_stack.push_back(m_previousBlock.hash);
			break;
		case Instruction::BLK_COINBASE:
			m_stack.push_back(m_currentBlock.coinbase);
			break;
		case Instruction::BLK_TIMESTAMP:
			m_stack.push_back(m_currentBlock.timestamp);
			break;
		case Instruction::BLK_NUMBER:
			m_stack.push_back(m_currentBlock.number);
			break;
		case Instruction::BLK_DIFFICULTY:
			m_stack.push_back(m_currentBlock.difficulty);
			break;
		case Instruction::SHA256:
		case Instruction::RIPEMD160:
		{
			uint s = (uint)min(m_stack.back(), (u256)(m_stack.size() - 1) * 32);
			bytes b(s);
			uint i = 0;
			for (; s; s = (s >= 32 ? s - 32 : 0), i += 32)
			{
				m_stack.pop_back();
				u256 v = m_stack.back();
				int sz = (int)min<u256>(32, s) - 1;						// sz is one fewer than the number of bytes we're interested in.
				v >>= ((31 - sz) * 8);									// kill unused low-order bytes.
				for (int j = 0; j <= sz; ++j, v >>= 8)					// cycle through bytes, starting at low-order end.
					b[i + sz - j] = (byte)(v & 0xff);					// set each 32-byte (256-bit) chunk in reverse - (i.e. we want to put low-order last).
			}
			if (inst == Instruction::SHA256)
				m_stack.back() = sha256(b);
			else
				// NOTE: this aligns to right of 256-bit container (low-order bytes). This won't work if they're treated as byte-arrays and thus left-aligned in a 256-bit container.
				m_stack.back() = ripemd160(&b);
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
			m_stack.push_back(mem(curPC + 1));
			nextPC = curPC + 2;
			break;
		}
		case Instruction::POP:
			require(1);
			m_stack.pop_back();
			break;
		case Instruction::DUP:
			require(1);
			m_stack.push_back(m_stack.back());
			break;
		case Instruction::DUPN:
		{
			auto s = mem(curPC + 1);
			if (s == 0 || s > m_stack.size())
				throw OperandOutOfRange(1, m_stack.size(), s);
			m_stack.push_back(m_stack[m_stack.size() - (uint)s]);
			nextPC = curPC + 2;
			break;
		}
		case Instruction::SWAP:
		{
			require(2);
			auto d = m_stack.back();
			m_stack.back() = m_stack[m_stack.size() - 2];
			m_stack[m_stack.size() - 2] = d;
			break;
		}
		case Instruction::SWAPN:
		{
			require(1);
			auto d = m_stack.back();
			auto s = mem(curPC + 1);
			if (s == 0 || s > m_stack.size())
				throw OperandOutOfRange(1, m_stack.size(), s);
			m_stack.back() = m_stack[m_stack.size() - (uint)s];
			m_stack[m_stack.size() - (uint)s] = d;
			nextPC = curPC + 2;
			break;
		}
		case Instruction::LOAD:
			require(1);
			m_stack.back() = mem(m_stack.back());
			break;
		case Instruction::STORE:
			require(2);
			mem(m_stack.back()) = m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			m_stack.pop_back();
			break;
		case Instruction::JMP:
			require(1);
			nextPC = m_stack.back();
			m_stack.pop_back();
			break;
		case Instruction::JMPI:
			require(2);
			if (m_stack.back())
				nextPC = m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			m_stack.pop_back();
			break;
		case Instruction::IND:
			m_stack.push_back(curPC);
			break;
		case Instruction::EXTRO:
		{
			require(2);
			auto memoryAddress = m_stack.back();
			m_stack.pop_back();
			auto contractAddress = m_stack.back();
			m_stack.back() = m_state->memory(contractAddress, memoryAddress);
			break;
		}
		case Instruction::BALANCE:
		{
			require(1);
			m_stack.back() = m_state->balance(m_stack.back());
			break;
		}
		case Instruction::MKTX:
		{
			require(4);
			auto dest = m_stack.back();
			m_stack.pop_back();

			auto amount = m_stack.back();
			m_stack.pop_back();

			auto fee = m_stack.back();
			m_stack.pop_back();

			auto itemCount = m_stack.back();
			m_stack.pop_back();
			if (m_stack.size() < itemCount)
				throw OperandOutOfRange(0, m_stack.size(), itemCount);
			u256s data;
			data.reserve((uint)itemCount);
			for (auto i = 0; i < itemCount; ++i)
			{
				data.push_back(m_stack.back());
				m_stack.pop_back();
			}
			m_state->transact(m_myAddress, dest, amount, fee, data);
			break;
		}
		case Instruction::SUICIDE:
			// TODO: Suicide...
		case Instruction::STOP:
			// TODO: Cleanups...
			return;
		default:
			throw BadInstruction();
		}
	}
}
