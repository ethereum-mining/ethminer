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
		case Instruction::MUL:
			//pops two items and pushes S[-1] * S[-2] mod 2^256.
			require(2);
			m_stack[m_stack.size() - 2] *= m_stack.back();
			m_stack.pop_back();
		case Instruction::SUB:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() - m_stack[m_stack.size() - 2];
			m_stack.pop_back();
		case Instruction::DIV:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() / m_stack[m_stack.size() - 2];
			m_stack.pop_back();
		case Instruction::SDIV:
			require(2);
			(s256&)m_stack[m_stack.size() - 2] = (s256&)m_stack.back() / (s256&)m_stack[m_stack.size() - 2];
			m_stack.pop_back();
		case Instruction::MOD:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() % m_stack[m_stack.size() - 2];
			m_stack.pop_back();
		case Instruction::SMOD:
			require(2);
			(s256&)m_stack[m_stack.size() - 2] = (s256&)m_stack.back() % (s256&)m_stack[m_stack.size() - 2];
			m_stack.pop_back();
		case Instruction::EXP:
			require(2);
//			(s256&)m_stack[m_stack.size() - 2] = pow(m_stack.back(), m_stack[m_stack.size() - 2]);
//			m_stack.pop_back();
		case Instruction::NEG:
		case Instruction::LT:
		case Instruction::LE:
		case Instruction::GT:
		case Instruction::GE:
		case Instruction::EQ:
		case Instruction::NOT:
		case Instruction::MYADDRESS:
		case Instruction::MYCREATOR:
		case Instruction::TXSENDER:
		case Instruction::TXVALUE:
		case Instruction::TXFEE:
		case Instruction::TXDATAN:
		case Instruction::TXDATA:
		case Instruction::BLK_PREVHASH:
		case Instruction::BLK_COINBASE:
		case Instruction::BLK_TIMESTAMP:
		case Instruction::BLK_NUMBER:
		case Instruction::BLK_DIFFICULTY:
		case Instruction::SHA256:
		case Instruction::RIPEMD160:
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
