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
#include <functional>
#include <libethcore/CommonEth.h>
#include <libevm/ExtVMFace.h>
#include "State.h"

namespace eth
{

/**
 * @brief Externality interface for the Virtual Machine providing access to world state.
 */
class ExtVM: public ExtVMFace
{
public:
	/// Full constructor.
	ExtVM(State& _s, Address _myAddress, Address _caller, Address _origin, u256 _value, u256 _gasPrice, bytesConstRef _data, bytesConstRef _code, Manifest* o_ms, unsigned _level = 0):
		ExtVMFace(_myAddress, _caller, _origin, _value, _gasPrice, _data, _code, _s.m_previousBlock, _s.m_currentBlock), level(_level), m_s(_s), m_origCache(_s.m_cache), m_ms(o_ms)
	{
		m_s.ensureCached(_myAddress, true, true);
	}

	/// Read storage location.
	u256 store(u256 _n) { return m_s.storage(myAddress, _n); }

	/// Write a value in storage.
	void setStore(u256 _n, u256 _v) { m_s.setStorage(myAddress, _n, _v); if (m_ms) m_ms->altered.push_back(_n); }

	/// Create a new contract.
	h160 create(u256 _endowment, u256* _gas, bytesConstRef _code, OnOpFunc const& _onOp = OnOpFunc())
	{
		// Increment associated nonce for sender.
		m_s.noteSending(myAddress);
		if (m_ms)
			m_ms->internal.resize(m_ms->internal.size() + 1);
		auto ret = m_s.create(myAddress, _endowment, gasPrice, _gas, _code, origin, &suicides, &posts, m_ms ? &(m_ms->internal.back()) : nullptr, _onOp, level + 1);
		if (m_ms && !m_ms->internal.back().from)
			m_ms->internal.pop_back();
		return ret;
	}

	/// Create a new message call. Leave _myAddressOverride at he default to use the present address as caller.
	bool call(Address _receiveAddress, u256 _txValue, bytesConstRef _txData, u256* _gas, bytesRef _out, OnOpFunc const& _onOp = OnOpFunc(), Address _myAddressOverride = Address())
	{
		if (m_ms)
			m_ms->internal.resize(m_ms->internal.size() + 1);
		auto ret = m_s.call(_receiveAddress, _myAddressOverride ? _myAddressOverride : myAddress, _txValue, gasPrice, _txData, _gas, _out, origin, &suicides, &posts, m_ms ? &(m_ms->internal.back()) : nullptr, _onOp, level + 1);
		if (m_ms && !m_ms->internal.back().from)
			m_ms->internal.pop_back();
		return ret;
	}

	/// Read address's balance.
	u256 balance(Address _a) { return m_s.balance(_a); }

	/// Subtract amount from account's balance.
	void subBalance(u256 _a) { m_s.subBalance(myAddress, _a); }

	/// Determine account's TX count.
	u256 txCount(Address _a) { return m_s.transactionsFrom(_a); }

	/// Suicide the associated contract to the given address.
	void suicide(Address _a)
	{
		m_s.addBalance(_a, m_s.balance(myAddress));
		ExtVMFace::suicide(_a);
	}

	/// Revert any changes made (by any of the other calls).
	/// @TODO check call site for the parent manifest being discarded.
	void revert() { if (m_ms) *m_ms = Manifest(); m_s.m_cache = m_origCache; }

	/// Execute any posts we have left.
	u256 doPosts(OnOpFunc const& _onOp = OnOpFunc())
	{
		u256 ret;
		while (posts.size())
		{
			Post& p = posts.front();
			call(p.to, p.value, &p.data, &p.gas, bytesRef(), _onOp, p.from);
			ret += p.gas;
			posts.pop_front();
		}
		return ret;
	}

	State& state() const { return m_s; }

	/// @note not a part of the main API; just for use by tracing/debug stuff.
	unsigned level = 0;

private:
	State& m_s;										///< A reference to the base state.
	std::map<Address, AddressState> m_origCache;	///< The cache of the address states (i.e. the externalities) as-was prior to the execution.
	Manifest* m_ms;
};

}


