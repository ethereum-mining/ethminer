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
/** @file ExtVM.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <map>
#include "CommonEth.h"
#include "State.h"
#include "ExtVMFace.h"

namespace eth
{

// TODO: Document
class ExtVM: public ExtVMFace
{
public:
	ExtVM(State& _s, Address _myAddress, Address _caller, Address _origin, u256 _value, u256 _gasPrice, bytesConstRef _data, bytesConstRef _code):
		ExtVMFace(_myAddress, _caller, _origin, _value, _gasPrice, _data, _code, _s.m_previousBlock, _s.m_currentBlock), m_s(_s), m_origCache(_s.m_cache)
	{
		m_s.ensureCached(_myAddress, true, true);
	}

	u256 store(u256 _n)
	{
		return m_s.storage(myAddress, _n);
	}
	void setStore(u256 _n, u256 _v)
	{
		m_s.setStorage(myAddress, _n, _v);
	}

	h160 create(u256 _endowment, u256* _gas, bytesConstRef _code)
	{
		// Increment associated nonce for sender.
		m_s.noteSending(myAddress);

		return m_s.create(myAddress, _endowment, gasPrice, _gas, _code, origin);
	}

	bool call(Address _receiveAddress, u256 _txValue, bytesConstRef _txData, u256* _gas, bytesRef _out)
	{
		return m_s.call(_receiveAddress, myAddress, _txValue, gasPrice, _txData, _gas, _out, origin);
	}

	u256 balance(Address _a) { return m_s.balance(_a); }
	void subBalance(u256 _a) { m_s.subBalance(myAddress, _a); }
	u256 txCount(Address _a) { return m_s.transactionsFrom(_a); }
	void suicide(Address _a)
	{
		m_s.addBalance(_a, m_s.balance(myAddress));
		m_s.m_cache[myAddress].kill();
	}

	void revert()
	{
		m_s.m_cache = m_origCache;
	}

private:
	State& m_s;
	std::map<Address, AddressState> m_origCache;
	std::map<u256, u256>* m_store;
};

}


