#include "sha256.h"
#include "VirtualMachine.h"
using namespace std;
using namespace eth;

VirtualMachine::VirtualMachine()
{
}

VirtualMachine::~VirtualMachine()
{
}


void VirtualMachine::go()
{
	bool stopped = false;
	while (!stopped)
	{
		auto rawInst = m_memory[m_pc];
		if (rawInst > 0xff)
			throw BadInstruction();
		Instruction inst = (Instruction)(uint8_t)rawInst;

		auto require = [&](uint _n)
		{
			if (m_stack.size() < _n)
				throw StackTooSmall(_n, m_stack.size());
		};

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
				m_stack.back() = ripemd160(&b);

			break;
		}
		case Instruction::ECMUL:
		case Instruction::ECADD:
		case Instruction::ECSIGN:
		case Instruction::ECRECOVER:
		case Instruction::ECVALID:
		case Instruction::PUSH:
		case Instruction::POP:
		case Instruction::DUP:
		case Instruction::DUPN:
		case Instruction::SWAP:
		case Instruction::SWAPN:
		case Instruction::LOAD:
		case Instruction::STORE:
		case Instruction::JMP:
		case Instruction::JMPI:
		case Instruction::IND:
		case Instruction::EXTRO:
		case Instruction::BALANCE:
		case Instruction::MKTX:
			break;
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
