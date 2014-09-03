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
/** @file Executive.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <boost/timer.hpp>
#include <libevm/VM.h>
#include "Executive.h"
#include "State.h"
#include "ExtVM.h"
using namespace std;
using namespace eth;

#define ETH_VMTRACE 1

Executive::~Executive()
{
	// TODO: Make safe.
	delete m_ext;
	delete m_vm;
}

u256 Executive::gasUsed() const
{
	return m_t.gas - m_endGas;
}

bool Executive::setup(bytesConstRef _rlp)
{
	// Entry point for a user-executed transaction.
	m_t = Transaction(_rlp);

	m_sender = m_t.sender();

	// Avoid invalid transactions.
	auto nonceReq = m_s.transactionsFrom(m_sender);
	if (m_t.nonce != nonceReq)
	{
		clog(StateChat) << "Invalid Nonce: Require" << nonceReq << " Got" << m_t.nonce;
		throw InvalidNonce(nonceReq, m_t.nonce);
	}

	// Don't like transactions whose gas price is too low. NOTE: this won't stay here forever - it's just until we get a proper gas price discovery protocol going.
	if (m_t.gasPrice < m_s.m_currentBlock.minGasPrice)
	{
		clog(StateChat) << "Offered gas-price is too low: Require >" << m_s.m_currentBlock.minGasPrice << " Got" << m_t.gasPrice;
		throw GasPriceTooLow();
	}

	// Check gas cost is enough.
	u256 gasCost = m_t.data.size() * c_txDataGas + c_txGas;

	if (m_t.gas < gasCost)
	{
		clog(StateChat) << "Not enough gas to pay for the transaction: Require >" << gasCost << " Got" << m_t.gas;
		throw OutOfGas();
	}

	u256 cost = m_t.value + m_t.gas * m_t.gasPrice;

	// Avoid unaffordable transactions.
	if (m_s.balance(m_sender) < cost)
	{
		clog(StateChat) << "Not enough cash: Require >" << cost << " Got" << m_s.balance(m_sender);
		throw NotEnoughCash();
	}

	u256 startGasUsed = m_s.gasUsed();
	if (startGasUsed + m_t.gas > m_s.m_currentBlock.gasLimit)
	{
		clog(StateChat) << "Too much gas used in this block: Require <" << (m_s.m_currentBlock.gasLimit - startGasUsed) << " Got" << m_t.gas;
		throw BlockGasLimitReached();
	}

	// Increment associated nonce for sender.
	m_s.noteSending(m_sender);

	// Pay...
//	cnote << "Paying" << formatBalance(cost) << "from sender (includes" << m_t.gas << "gas at" << formatBalance(m_t.gasPrice) << ")";
	m_s.subBalance(m_sender, cost);

	if (m_ms)
	{
		m_ms->from = m_sender;
		m_ms->to = m_t.receiveAddress;
		m_ms->value = m_t.value;
		m_ms->input = m_t.data;
	}

	if (m_t.isCreation())
		return create(m_sender, m_t.value, m_t.gasPrice, m_t.gas - gasCost, &m_t.data, m_sender);
	else
		return call(m_t.receiveAddress, m_sender, m_t.value, m_t.gasPrice, bytesConstRef(&m_t.data), m_t.gas - gasCost, m_sender);
}

bool Executive::call(Address _receiveAddress, Address _senderAddress, u256 _value, u256 _gasPrice, bytesConstRef _data, u256 _gas, Address _originAddress)
{
//	cnote << "Transferring" << formatBalance(_value) << "to receiver.";
	m_s.addBalance(_receiveAddress, _value);

	if (m_s.addressHasCode(_receiveAddress))
	{
		m_vm = new VM(_gas);
		bytes const& c = m_s.code(_receiveAddress);
		m_ext = new ExtVM(m_s, _receiveAddress, _senderAddress, _originAddress, _value, _gasPrice, _data, &c, m_ms);
	}
	else
		m_endGas = _gas;
	return !m_ext;
}

bool Executive::create(Address _sender, u256 _endowment, u256 _gasPrice, u256 _gas, bytesConstRef _init, Address _origin)
{
	// We can allow for the reverted state (i.e. that with which m_ext is constructed) to contain the m_newAddress, since
	// we delete it explicitly if we decide we need to revert.
	m_newAddress = right160(sha3(rlpList(_sender, m_s.transactionsFrom(_sender) - 1)));
	while (m_s.addressInUse(m_newAddress))
		m_newAddress = (u160)m_newAddress + 1;

	// Set up new account...
	m_s.m_cache[m_newAddress] = AddressState(0, _endowment, h256(), h256());

	// Execute _init.
	m_vm = new VM(_gas);
	m_ext = new ExtVM(m_s, m_newAddress, _sender, _origin, _endowment, _gasPrice, bytesConstRef(), _init, m_ms);
	return _init.empty();
}

OnOpFunc Executive::simpleTrace()
{
	return [](uint64_t steps, Instruction inst, bigint newMemSize, bigint gasCost, void* voidVM, void const* voidExt)
	{
		ExtVM const& ext = *(ExtVM const*)voidExt;
		VM& vm = *(VM*)voidVM;

		ostringstream o;
		o << endl << "    STACK" << endl;
		for (auto i: vm.stack())
			o << (h256)i << endl;
		o << "    MEMORY" << endl << memDump(vm.memory());
		o << "    STORAGE" << endl;
		for (auto const& i: ext.state().storage(ext.myAddress))
			o << showbase << hex << i.first << ": " << i.second << endl;
		eth::LogOutputStream<VMTraceChannel, false>(true) << o.str();
		eth::LogOutputStream<VMTraceChannel, false>(false) << " | " << dec << ext.level << " | " << ext.myAddress << " | #" << steps << " | " << hex << setw(4) << setfill('0') << vm.curPC() << " : " << instructionInfo(inst).name << " | " << dec << vm.gas() << " | -" << dec << gasCost << " | " << newMemSize << "x32" << " ]";
	};
}

bool Executive::go(OnOpFunc const& _onOp)
{
	if (m_vm)
	{
		boost::timer t;
		auto sgas = m_vm->gas();
		bool revert = false;
		try
		{
			m_out = m_vm->go(*m_ext, _onOp);
			m_endGas = m_vm->gas();
		}
		catch (StepsDone const&)
		{
			return false;
		}
		catch (OutOfGas const& /*_e*/)
		{
			clog(StateChat) << "Out of Gas! Reverting.";
			revert = true;
		}
		catch (VMException const& _e)
		{
			clog(StateChat) << "VM Exception: " << _e.description();
		}
		catch (Exception const& _e)
		{
			clog(StateChat) << "Exception in VM: " << _e.description();
		}
		catch (std::exception const& _e)
		{
			clog(StateChat) << "std::exception in VM: " << _e.what();
		}
		cnote << "VM took:" << t.elapsed() << "; gas used: " << (sgas - m_endGas);

		// Write state out only in the case of a non-excepted transaction.
		if (revert)
		{
			m_ext->revert();
			// Explicitly delete a newly created address - this will still be in the reverted state.
			if (m_newAddress)
			{
				m_s.m_cache.erase(m_newAddress);
				m_newAddress = Address();
			}
		}
	}
	return true;
}

u256 Executive::gas() const
{
	return m_vm ? m_vm->gas() : m_endGas;
}

void Executive::finalize(OnOpFunc const& _onOp)
{
	if (m_t.isCreation() && m_newAddress && m_out.size())
		// non-reverted creation - put code in place.
		m_s.m_cache[m_newAddress].setCode(m_out);

	if (m_ext)
		m_endGas += m_ext->doPosts(_onOp);

//	cnote << "Refunding" << formatBalance(m_endGas * m_ext->gasPrice) << "to origin (=" << m_endGas << "*" << formatBalance(m_ext->gasPrice) << ")";
	m_s.addBalance(m_sender, m_endGas * m_t.gasPrice);

	u256 feesEarned = (m_t.gas - m_endGas) * m_t.gasPrice;
//	cnote << "Transferring" << formatBalance(gasSpent) << "to miner.";
	m_s.addBalance(m_s.m_currentBlock.coinbaseAddress, feesEarned);

	if (m_ms)
		m_ms->output = m_out.toBytes();

	// Suicides...
	if (m_ext)
		for (auto a: m_ext->suicides)
			m_s.m_cache[a].kill();
}
