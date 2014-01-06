/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file State.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <secp256k1.h>
#include <random>
#include "Trie.h"
#include "BlockChain.h"
#include "Instruction.h"
#include "Exceptions.h"
#include "sha256.h"
#include "State.h"
using namespace std;
using namespace eth;

u256 const State::c_stepFee = 0;
u256 const State::c_dataFee = 0;
u256 const State::c_memoryFee = 0;
u256 const State::c_extroFee = 0;
u256 const State::c_cryptoFee = 0;
u256 const State::c_newContractFee = 0;
u256 const State::c_txFee = 0;

State::State(Address _minerAddress): m_minerAddress(_minerAddress)
{
	secp256k1_start();
	// TODO: Initialise current block/previous block, ready for sync.
}

void State::sync(BlockChain const& _bc, TransactionQueue const& _tq)
{
	BlockInfo bi;
	try
	{
		bi.verify(_bc.lastBlock(), _bc.lastBlockNumber());
	}
	catch (...)
	{
		cerr << "ERROR: Corrupt block-chain! Delete your block-chain DB and restart." << endl;
		exit(1);
	}

	if (bi == m_currentBlock)
	{
		// We mined the last block.
		// Our state is good - we just need to move on to next.
		m_previousBlock = m_currentBlock;
		m_current.clear();
		m_transactions.clear();
		m_currentBlock = BlockInfo();
		m_currentBlock.number = m_previousBlock.number + 1;
	}
	else if (bi == m_previousBlock)
	{
		// No change since last sync.
		// Carry on as we were.
	}
	else
	{
		// New blocks available, or we've switched to a different branch. All change.
		// TODO: Find most recent state dump and replay what's left.
		// (Most recent state dump might end up being genesis.)
	}
}

bool State::mine(uint _msTimeout) const
{
	// TODO: update timestamp according to clock.
	// TODO: update difficulty according to timestamp.
	// TODO: look for a nonce that makes a good hash.
	// ...but don't take longer than _msTimeout ms.
	return false;
}

bool State::isNormalAddress(Address _address) const
{
	auto it = m_current.find(_address);
	return it != m_current.end() && it->second.type() == AddressType::Normal;
}

bool State::isContractAddress(Address _address) const
{
	auto it = m_current.find(_address);
	return it != m_current.end() && it->second.type() == AddressType::Contract;
}

u256 State::balance(Address _id) const
{
	auto it = m_current.find(_id);
	return it == m_current.end() ? 0 : it->second.balance();
}

void State::addBalance(Address _id, u256 _amount)
{
	auto it = m_current.find(_id);
	if (it == m_current.end())
		it->second.balance() = _amount;
	else
		it->second.balance() += _amount;
}

void State::subBalance(Address _id, bigint _amount)
{
	auto it = m_current.find(_id);
	if (it == m_current.end() || (bigint)it->second.balance() < _amount)
		throw NotEnoughCash();
	it->second.balance() = (u256)((bigint)it->second.balance() - _amount);
}

u256 State::transactionsFrom(Address _address) const
{
	auto it = m_current.find(_address);
	return it == m_current.end() ? 0 : it->second.nonce();
}

u256 State::contractMemory(Address _contract, u256 _memory) const
{
	auto m = m_current.find(_contract);
	if (m == m_current.end())
		return 0;
	auto i = m->second.memory().find(_memory);
	return i == m->second.memory().end() ? 0 : i->second;
}

bool State::verify(bytes const& _block, uint _number)
{
	BlockInfo bi;
	try
	{
		bi.verify(bytesConstRef((bytes*)&_block), _number);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

void State::execute(Transaction const& _t, Address _sender)
{
	// Entry point for a contract-originated transaction.
	m_transactions.push_back(_t);

	if (_t.nonce != transactionsFrom(_sender))
		throw InvalidNonce();

	if (balance(_sender) < _t.value + _t.fee)
		throw NotEnoughCash();

	if (_t.receiveAddress)
	{
		subBalance(_sender, _t.value + _t.fee);
		addBalance(_t.receiveAddress, _t.value);
		addBalance(m_minerAddress, _t.fee);

		if (isContractAddress(_t.receiveAddress))
		{
			MinerFeeAdder feeAdder({this, 0});	// will add fee on destruction.
			execute(_t.receiveAddress, _sender, _t.value, _t.fee, _t.data, &feeAdder.fee);
		}
	}
	else
	{
		if (_t.fee < _t.data.size() * c_memoryFee + c_newContractFee)
			throw FeeTooSmall();

		Address newAddress = low160(_t.sha256());

		if (isContractAddress(newAddress))
			throw ContractAddressCollision();

		auto& mem = m_current[newAddress].memory();
		for (uint i = 0; i < _t.data.size(); ++i)
			mem[i] = _t.data[i];
		subBalance(_sender, _t.value + _t.fee);
		addBalance(newAddress, _t.value);
		addBalance(m_minerAddress, _t.fee);
	}
}

void State::execute(Address _myAddress, Address _txSender, u256 _txValue, u256 _txFee, u256s const& _txData, u256* _totalFee)
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

		case Instruction::MKTX:
			minerFee += c_txFee;
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
			throw NotEnoughCash();
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
		{
			// ECMUL - pops three items.
			// If (S[-2],S[-1]) are a valid point in secp256k1, including both coordinates being less than P, pushes (S[-1],S[-2]) * S[-3], using (0,0) as the point at infinity.
			// Otherwise, pushes (0,0).
			require(3);

			bytes pub(1, 4);
			pub += toBigEndian(stack[stack.size() - 2]);
			pub += toBigEndian(stack.back());
			stack.pop_back();
			stack.pop_back();

			bytes x = toBigEndian(stack.back());
			stack.pop_back();

			if (secp256k1_ecdsa_pubkey_verify(pub.data(), pub.size()))	// TODO: Check both are less than P.
			{
				secp256k1_ecdsa_pubkey_tweak_mul(pub.data(), pub.size(), x.data());
				stack.push_back(fromBigEndian<u256>(bytesConstRef(&pub).cropped(1, 32)));
				stack.push_back(fromBigEndian<u256>(bytesConstRef(&pub).cropped(33, 32)));
			}
			else
			{
				stack.push_back(0);
				stack.push_back(0);
			}
			break;
		}
		case Instruction::ECADD:
		{
			// ECADD - pops four items and pushes (S[-4],S[-3]) + (S[-2],S[-1]) if both points are valid, otherwise (0,0).
			require(4);

			bytes pub(1, 4);
			pub += toBigEndian(stack[stack.size() - 2]);
			pub += toBigEndian(stack.back());
			stack.pop_back();
			stack.pop_back();

			bytes tweak(1, 4);
			tweak += toBigEndian(stack[stack.size() - 2]);
			tweak += toBigEndian(stack.back());
			stack.pop_back();
			stack.pop_back();

			if (secp256k1_ecdsa_pubkey_verify(pub.data(), pub.size()) && secp256k1_ecdsa_pubkey_verify(tweak.data(), tweak.size()))
			{
				secp256k1_ecdsa_pubkey_tweak_add(pub.data(), pub.size(), tweak.data());
				stack.push_back(fromBigEndian<u256>(bytesConstRef(&pub).cropped(1, 32)));
				stack.push_back(fromBigEndian<u256>(bytesConstRef(&pub).cropped(33, 32)));
			}
			else
			{
				stack.push_back(0);
				stack.push_back(0);
			}
			break;
		}
		case Instruction::ECSIGN:
		{
			require(2);
			bytes sig(64);
			int v = 0;

			u256 msg = stack.back();
			stack.pop_back();
			u256 priv = stack.back();
			stack.pop_back();
			bytes nonce = toBigEndian(Transaction::kFromMessage(msg, priv));

			if (!secp256k1_ecdsa_sign_compact(toBigEndian(msg).data(), 64, sig.data(), toBigEndian(priv).data(), nonce.data(), &v))
				throw InvalidSignature();

			stack.push_back(v + 27);
			stack.push_back(fromBigEndian<u256>(bytesConstRef(&sig).cropped(0, 32)));
			stack.push_back(fromBigEndian<u256>(bytesConstRef(&sig).cropped(32)));
			break;
		}
		case Instruction::ECRECOVER:
		{
			require(4);

			bytes sig = toBigEndian(stack[stack.size() - 2]) + toBigEndian(stack.back());
			stack.pop_back();
			stack.pop_back();
			int v = (int)stack.back();
			stack.pop_back();
			bytes msg = toBigEndian(stack.back());
			stack.pop_back();

			byte pubkey[65];
			int pubkeylen = 65;
			if (secp256k1_ecdsa_recover_compact(msg.data(), msg.size(), sig.data(), pubkey, &pubkeylen, 0, v - 27))
			{
				stack.push_back(0);
				stack.push_back(0);
			}
			else
			{
				stack.push_back(fromBigEndian<u256>(bytesConstRef(&pubkey[1], 32)));
				stack.push_back(fromBigEndian<u256>(bytesConstRef(&pubkey[33], 32)));
			}
			break;
		}
		case Instruction::ECVALID:
		{
			require(2);
			bytes pub(1, 4);
			pub += toBigEndian(stack[stack.size() - 2]);
			pub += toBigEndian(stack.back());
			stack.pop_back();
			stack.pop_back();

			stack.back() = secp256k1_ecdsa_pubkey_verify(pub.data(), pub.size()) ? 1 : 0;
			break;
		}
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
			Address contractAddress = as160(stack.back());
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
			execute(t, _myAddress);

			break;
		}
		case Instruction::SUICIDE:
		{
			require(1);
			Address dest = as160(stack.back());
			u256 minusVoidFee = m_current[_myAddress].memory().size() * c_memoryFee;
			addBalance(dest, balance(_myAddress) + minusVoidFee);
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
