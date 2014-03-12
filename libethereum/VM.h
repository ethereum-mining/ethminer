/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file VM.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <unordered_map>
#include "CryptoHeaders.h"
#include "Common.h"
#include "Exceptions.h"
#include "FeeStructure.h"
#include "Instruction.h"
#include "BlockInfo.h"
#include "ExtVMFace.h"

namespace eth
{

// Convert from a 256-bit integer stack/memory entry into a 160-bit Address hash.
// Currently we just pull out the right (low-order in BE) 160-bits.
inline Address asAddress(u256 _item)
{
	return right160(h256(_item));
}

inline u256 fromAddress(Address _a)
{
	return (u160)_a;
}

/**
 */
class VM
{
	template <unsigned T> friend class UnitTest;

public:
	/// Construct VM object.
	VM();

	void reset();

	template <class Ext>
	void go(Ext& _ext, uint64_t _steps = (uint64_t)-1);

	void require(u256 _n) { if (m_stack.size() < _n) throw StackTooSmall(_n, m_stack.size()); }
	u256 runFee() const { return m_runFee; }

private:
	u256 m_curPC = 0;
	u256 m_nextPC = 1;
	uint64_t m_stepCount = 0;
	std::map<u256, u256> m_temp;
	std::vector<u256> m_stack;
	u256 m_runFee = 0;
};

}

// INLINE:
template <class Ext> void eth::VM::go(Ext& _ext, uint64_t _steps)
{
	for (bool stopped = false; !stopped && _steps--; m_curPC = m_nextPC, m_nextPC = m_curPC + 1)
	{
		m_stepCount++;

		// INSTRUCTION...
		auto rawInst = _ext.store(m_curPC);
		if (rawInst > 0xff)
			throw BadInstruction();
		Instruction inst = (Instruction)(uint8_t)rawInst;

		// FEES...
		bigint runFee = m_stepCount > 16 ? _ext.fees.m_stepFee : 0;
		bigint storeCostDelta = 0;
		switch (inst)
		{
		case Instruction::SSTORE:
			require(2);
			if (!_ext.store(m_stack.back()) && m_stack[m_stack.size() - 2])
				storeCostDelta += _ext.fees.m_memoryFee;
			if (_ext.store(m_stack.back()) && !m_stack[m_stack.size() - 2])
				storeCostDelta -= _ext.fees.m_memoryFee;
			// continue on to...
		case Instruction::SLOAD:
			runFee += _ext.fees.m_dataFee;
			break;

		case Instruction::BALANCE:
			runFee += _ext.fees.m_extroFee;
			break;

		case Instruction::CALL:
			runFee += _ext.fees.m_txFee;
			break;

		default:
			break;
		}
		// TODO: payFee should deduct from origin.
		_ext.payFee(runFee + storeCostDelta);
		m_runFee += (u256)runFee;

		// EXECUTE...
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
			if (!m_stack[m_stack.size() - 2])
				return;
			m_stack[m_stack.size() - 2] = m_stack.back() / m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			break;
		case Instruction::SDIV:
			require(2);
			if (!m_stack[m_stack.size() - 2])
				return;
			(s256&)m_stack[m_stack.size() - 2] = (s256&)m_stack.back() / (s256&)m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			break;
		case Instruction::MOD:
			require(2);
			if (!m_stack[m_stack.size() - 2])
				return;
			m_stack[m_stack.size() - 2] = m_stack.back() % m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			break;
		case Instruction::SMOD:
			require(2);
			if (!m_stack[m_stack.size() - 2])
				return;
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
		case Instruction::GT:
			require(2);
			m_stack[m_stack.size() - 2] = m_stack.back() > m_stack[m_stack.size() - 2] ? 1 : 0;
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
			break;
		case Instruction::SHA3:
		{
			require(1);
			uint s = (uint)std::min(m_stack.back(), (u256)(m_stack.size() - 1) * 32);
			m_stack.pop_back();

			CryptoPP::SHA3_256 digest;
			uint i = 0;
			for (; s; s = (s >= 32 ? s - 32 : 0), i += 32)
			{
				bytes b = toBigEndian(m_stack.back());
				digest.Update(b.data(), (int)std::min<u256>(32, s));			// b.size() == 32
				m_stack.pop_back();
			}
			std::array<byte, 32> final;
			digest.TruncatedFinal(final.data(), 32);
			m_stack.push_back(fromBigEndian<u256>(final));
			break;
		}
		case Instruction::ADDRESS:
			m_stack.push_back(fromAddress(_ext.myAddress));
			break;
		case Instruction::ORIGIN:
			// TODO get originator from ext.
			m_stack.push_back(fromAddress(_ext.txSender));
			break;
		case Instruction::BALANCE:
		{
			require(1);
			m_stack.back() = _ext.balance(asAddress(m_stack.back()));
			break;
		}
		case Instruction::CALLER:
			m_stack.push_back(fromAddress(_ext.txSender));
			break;
		case Instruction::CALLVALUE:
			m_stack.push_back(_ext.txValue);
			break;
		case Instruction::CALLDATA:
			// TODO: write data from ext into memory.
			break;
		case Instruction::CALLDATASIZE:
			m_stack.push_back(_ext.txData.size());
			break;
		case Instruction::BASEFEE:
			m_stack.push_back(_ext.fees.multiplier());
			break;
		case Instruction::PREVHASH:
			m_stack.push_back(_ext.previousBlock.hash);
			break;
		case Instruction::PREVNONCE:
			m_stack.push_back(_ext.previousBlock.nonce);
			break;
		case Instruction::COINBASE:
			m_stack.push_back((u160)_ext.currentBlock.coinbaseAddress);
			break;
		case Instruction::TIMESTAMP:
			m_stack.push_back(_ext.currentBlock.timestamp);
			break;
		case Instruction::NUMBER:
			m_stack.push_back(_ext.currentNumber);
			break;
		case Instruction::DIFFICULTY:
			m_stack.push_back(_ext.currentBlock.difficulty);
			break;
		case Instruction::PUSH:
		{
			m_stack.push_back(_ext.store(m_curPC + 1));
			m_nextPC = m_curPC + 2;
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
		/*case Instruction::DUPN:
		{
			auto s = store(curPC + 1);
			if (s == 0 || s > stack.size())
				throw OperandOutOfRange(1, stack.size(), s);
			stack.push_back(stack[stack.size() - (uint)s]);
			nextPC = curPC + 2;
			break;
		}*/
		case Instruction::SWAP:
		{
			require(2);
			auto d = m_stack.back();
			m_stack.back() = m_stack[m_stack.size() - 2];
			m_stack[m_stack.size() - 2] = d;
			break;
		}
		/*case Instruction::SWAPN:
		{
			require(1);
			auto d = stack.back();
			auto s = store(curPC + 1);
			if (s == 0 || s > stack.size())
				throw OperandOutOfRange(1, stack.size(), s);
			stack.back() = stack[stack.size() - (uint)s];
			stack[stack.size() - (uint)s] = d;
			nextPC = curPC + 2;
			break;
		}*/
		case Instruction::MLOAD:
		{
			require(1);
			m_stack.back() = m_temp[m_stack.back()];
			break;
		}
		case Instruction::MSTORE:
		{
			require(2);
			m_temp[m_stack.back()] = m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			m_stack.pop_back();
			break;
		}
		case Instruction::MSTORE8:
		{
			require(2);
			m_temp[m_stack.back()] = m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			m_stack.pop_back();
			break;
		}
		case Instruction::SLOAD:
			require(1);
			m_stack.back() = _ext.store(m_stack.back());
			break;
		case Instruction::SSTORE:
			require(2);
			_ext.setStore(m_stack.back(), m_stack[m_stack.size() - 2]);
			m_stack.pop_back();
			m_stack.pop_back();
			break;
		case Instruction::JUMP:
			require(1);
			m_nextPC = m_stack.back();
			m_stack.pop_back();
			break;
		case Instruction::JUMPI:
			require(2);
			if (m_stack.back())
				m_nextPC = m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			m_stack.pop_back();
			break;
		case Instruction::PC:
			m_stack.push_back(m_curPC);
			break;
		case Instruction::CALL:
		{
			require(6);

			Transaction t;
			t.receiveAddress = asAddress(m_stack.back());
			m_stack.pop_back();
			t.value = m_stack.back();
			m_stack.pop_back();

			auto itemCount = m_stack.back();
			m_stack.pop_back();
			if (m_stack.size() < itemCount)
				throw OperandOutOfRange(0, m_stack.size(), itemCount);
			t.data.reserve((uint)itemCount);
			for (auto i = 0; i < itemCount; ++i)
			{
				t.data.push_back(m_stack.back());
				m_stack.pop_back();
			}

			_ext.mktx(t);
			break;
		}
		case Instruction::RETURN:
			require(2);
			// TODO: write data from memory into ext.
			return;
		case Instruction::SUICIDE:
		{
			require(1);
			Address dest = asAddress(m_stack.back());
			_ext.suicide(dest);
			// ...follow through to...
		}
		case Instruction::STOP:
			return;
		default:
			throw BadInstruction();
		}
	}
	if (_steps == (unsigned)-1)
		throw StepsDone();
}

