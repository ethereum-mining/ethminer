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
#include <secp256k1.h>
#if WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#else
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include <sha.h>
#include <sha3.h>
#include <ripemd.h>
#if WIN32
#pragma warning(pop)
#else
#endif
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

		case Instruction::EXTRO:
		case Instruction::BALANCE:
			runFee += _ext.fees.m_extroFee;
			break;

		case Instruction::MKTX:
			runFee += _ext.fees.m_txFee;
			break;

		case Instruction::SHA256:
		case Instruction::RIPEMD160:
		case Instruction::ECMUL:
		case Instruction::ECADD:
		case Instruction::ECSIGN:
		case Instruction::ECRECOVER:
		case Instruction::ECVALID:
			runFee += _ext.fees.m_cryptoFee;
			break;
		default:
			break;
		}
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
			break;
		case Instruction::MYADDRESS:
			m_stack.push_back(fromAddress(_ext.myAddress));
			break;
		case Instruction::TXSENDER:
			m_stack.push_back(fromAddress(_ext.txSender));
			break;
		case Instruction::TXVALUE:
			m_stack.push_back(_ext.txValue);
			break;
		case Instruction::TXDATAN:
			m_stack.push_back(_ext.txData.size());
			break;
		case Instruction::TXDATA:
			require(1);
			m_stack.back() = m_stack.back() < _ext.txData.size() ? _ext.txData[(uint)m_stack.back()] : 0;
			break;
		case Instruction::BLK_PREVHASH:
			m_stack.push_back(_ext.previousBlock.hash);
			break;
		case Instruction::BLK_COINBASE:
			m_stack.push_back((u160)_ext.currentBlock.coinbaseAddress);
			break;
		case Instruction::BLK_TIMESTAMP:
			m_stack.push_back(_ext.currentBlock.timestamp);
			break;
		case Instruction::BLK_NUMBER:
			m_stack.push_back(_ext.currentNumber);
			break;
		case Instruction::BLK_DIFFICULTY:
			m_stack.push_back(_ext.currentBlock.difficulty);
			break;
		case Instruction::BLK_NONCE:
			m_stack.push_back(_ext.previousBlock.nonce);
			break;
		case Instruction::BASEFEE:
			m_stack.push_back(_ext.fees.multiplier());
			break;
		case Instruction::SHA256:
		{
			require(1);
			uint s = (uint)std::min(m_stack.back(), (u256)(m_stack.size() - 1) * 32);
			m_stack.pop_back();

			CryptoPP::SHA256 digest;
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
		case Instruction::RIPEMD160:
		{
			require(1);
			uint s = (uint)std::min(m_stack.back(), (u256)(m_stack.size() - 1) * 32);
			m_stack.pop_back();

			CryptoPP::RIPEMD160 digest;
			uint i = 0;
			for (; s; s = (s >= 32 ? s - 32 : 0), i += 32)
			{
				bytes b = toBigEndian(m_stack.back());
				digest.Update(b.data(), (int)std::min<u256>(32, s));			// b.size() == 32
				m_stack.pop_back();
			}
			std::array<byte, 20> final;
			digest.TruncatedFinal(final.data(), 20);
			// NOTE: this aligns to right of 256-bit container (low-order bytes).
			// This won't work if they're treated as byte-arrays and thus left-aligned in a 256-bit container.
			m_stack.push_back((u256)fromBigEndian<u160>(final));
			break;
		}
		case Instruction::ECMUL:
		{
			// ECMUL - pops three items.
			// If (S[-2],S[-1]) are a valid point in secp256k1, including both coordinates being less than P, pushes (S[-1],S[-2]) * S[-3], using (0,0) as the point at infinity.
			// Otherwise, pushes (0,0).
			require(3);

			bytes pub(1, 4);
			pub += toBigEndian(m_stack[m_stack.size() - 2]);
			pub += toBigEndian(m_stack.back());
			m_stack.pop_back();
			m_stack.pop_back();

			bytes x = toBigEndian(m_stack.back());
			m_stack.pop_back();

			if (secp256k1_ecdsa_pubkey_verify(pub.data(), (int)pub.size()))	// TODO: Check both are less than P.
			{
				secp256k1_ecdsa_pubkey_tweak_mul(pub.data(), (int)pub.size(), x.data());
				m_stack.push_back(fromBigEndian<u256>(bytesConstRef(&pub).cropped(1, 32)));
				m_stack.push_back(fromBigEndian<u256>(bytesConstRef(&pub).cropped(33, 32)));
			}
			else
			{
				m_stack.push_back(0);
				m_stack.push_back(0);
			}
			break;
		}
		case Instruction::ECADD:
		{
			// ECADD - pops four items and pushes (S[-4],S[-3]) + (S[-2],S[-1]) if both points are valid, otherwise (0,0).
			require(4);

			bytes pub(1, 4);
			pub += toBigEndian(m_stack[m_stack.size() - 2]);
			pub += toBigEndian(m_stack.back());
			m_stack.pop_back();
			m_stack.pop_back();

			bytes tweak(1, 4);
			tweak += toBigEndian(m_stack[m_stack.size() - 2]);
			tweak += toBigEndian(m_stack.back());
			m_stack.pop_back();
			m_stack.pop_back();

			if (secp256k1_ecdsa_pubkey_verify(pub.data(),(int) pub.size()) && secp256k1_ecdsa_pubkey_verify(tweak.data(),(int) tweak.size()))
			{
				secp256k1_ecdsa_pubkey_tweak_add(pub.data(), (int)pub.size(), tweak.data());
				m_stack.push_back(fromBigEndian<u256>(bytesConstRef(&pub).cropped(1, 32)));
				m_stack.push_back(fromBigEndian<u256>(bytesConstRef(&pub).cropped(33, 32)));
			}
			else
			{
				m_stack.push_back(0);
				m_stack.push_back(0);
			}
			break;
		}
		case Instruction::ECSIGN:
		{
			require(2);
			bytes sig(64);
			int v = 0;

			u256 msg = m_stack.back();
			m_stack.pop_back();
			u256 priv = m_stack.back();
			m_stack.pop_back();
			bytes nonce = toBigEndian(Transaction::kFromMessage(msg, priv));

			if (!secp256k1_ecdsa_sign_compact(toBigEndian(msg).data(), 64, sig.data(), toBigEndian(priv).data(), nonce.data(), &v))
				throw InvalidSignature();

			m_stack.push_back(v + 27);
			m_stack.push_back(fromBigEndian<u256>(bytesConstRef(&sig).cropped(0, 32)));
			m_stack.push_back(fromBigEndian<u256>(bytesConstRef(&sig).cropped(32)));
			break;
		}
		case Instruction::ECRECOVER:
		{
			require(4);

			bytes sig = toBigEndian(m_stack[m_stack.size() - 2]) + toBigEndian(m_stack.back());
			m_stack.pop_back();
			m_stack.pop_back();
			int v = (int)m_stack.back();
			m_stack.pop_back();
			bytes msg = toBigEndian(m_stack.back());
			m_stack.pop_back();

			byte pubkey[65];
			int pubkeylen = 65;
			if (secp256k1_ecdsa_recover_compact(msg.data(), (int)msg.size(), sig.data(), pubkey, &pubkeylen, 0, v - 27))
			{
				m_stack.push_back(0);
				m_stack.push_back(0);
			}
			else
			{
				m_stack.push_back(fromBigEndian<u256>(bytesConstRef(&pubkey[1], 32)));
				m_stack.push_back(fromBigEndian<u256>(bytesConstRef(&pubkey[33], 32)));
			}
			break;
		}
		case Instruction::ECVALID:
		{
			require(2);
			bytes pub(1, 4);
			pub += toBigEndian(m_stack[m_stack.size() - 2]);
			pub += toBigEndian(m_stack.back());
			m_stack.pop_back();
			m_stack.pop_back();

			m_stack.back() = secp256k1_ecdsa_pubkey_verify(pub.data(), (int)pub.size()) ? 1 : 0;
			break;
		}
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
#ifdef __clang__
			auto mFinder = m_temp.find(m_stack.back());
			if (mFinder != m_temp.end())
				m_stack.back() = mFinder->second;
			else
				m_stack.back() = 0;
#else
			m_stack.back() = m_temp[m_stack.back()];
#endif
			break;
		}
		case Instruction::MSTORE:
		{
			require(2);
#ifdef __clang__
			auto mFinder = m_temp.find(m_stack.back());
			if (mFinder == m_temp.end())
				m_temp.insert(std::make_pair(m_stack.back(), m_stack[m_stack.size() - 2]));
			else
				mFinder->second = m_stack[m_stack.size() - 2];
#else
			m_temp[m_stack.back()] = m_stack[m_stack.size() - 2];
#endif
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
		case Instruction::JMP:
			require(1);
			m_nextPC = m_stack.back();
			m_stack.pop_back();
			break;
		case Instruction::JMPI:
			require(2);
			if (m_stack.back())
				m_nextPC = m_stack[m_stack.size() - 2];
			m_stack.pop_back();
			m_stack.pop_back();
			break;
		case Instruction::IND:
			m_stack.push_back(m_curPC);
			break;
		case Instruction::EXTRO:
		{
			require(2);
			auto memoryAddress = m_stack.back();
			m_stack.pop_back();
			Address contractAddress = asAddress(m_stack.back());
			m_stack.back() = _ext.extro(contractAddress, memoryAddress);
			break;
		}
		case Instruction::BALANCE:
		{
			require(1);
			m_stack.back() = _ext.balance(asAddress(m_stack.back()));
			break;
		}
		case Instruction::MKTX:
		{
			require(3);

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

