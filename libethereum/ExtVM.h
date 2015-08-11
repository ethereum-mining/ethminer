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
#include <libethcore/Common.h>
#include <libevm/ExtVMFace.h>
#include "State.h"

namespace dev
{
namespace eth
{

/**
 * @brief Externality interface for the Virtual Machine providing access to world state.
 */
class ExtVM: public ExtVMFace
{
public:
	/// Full constructor.
	ExtVM(State& _s, EnvInfo const& _envInfo, Address _myAddress, Address _caller, Address _origin, u256 _value, u256 _gasPrice, bytesConstRef _data, bytesConstRef _code, h256 const& _codeHash, unsigned _depth = 0):
		ExtVMFace(_envInfo, _myAddress, _caller, _origin, _value, _gasPrice, _data, _code.toBytes(), _codeHash, _depth), m_s(_s), m_origCache(_s.m_cache)
	{
		m_s.ensureCached(_myAddress, true, true);
	}

	/// Read storage location.
	virtual u256 store(u256 _n) override final { return m_s.storage(myAddress, _n); }

	/// Write a value in storage.
	virtual void setStore(u256 _n, u256 _v) override final { m_s.setStorage(myAddress, _n, _v); }

	/// Read address's code.
	virtual bytes const& codeAt(Address _a) override final { return m_s.code(_a); }

	/// Create a new contract.
	virtual h160 create(u256 _endowment, u256& io_gas, bytesConstRef _code, OnOpFunc const& _onOp = {}) override final;

	/// Create a new message call. Leave _myAddressOverride as the default to use the present address as caller.
	virtual bool call(CallParameters& _params) override final;

	/// Read address's balance.
	virtual u256 balance(Address _a) override final { return m_s.balance(_a); }

	/// Subtract amount from account's balance.
	virtual void subBalance(u256 _a) override final { m_s.subBalance(myAddress, _a); }

	/// Determine account's TX count.
	virtual u256 txCount(Address _a) override final { return m_s.transactionsFrom(_a); }

	/// Does the account exist?
	virtual bool exists(Address _a) override final { return m_s.addressInUse(_a); }

	/// Suicide the associated contract to the given address.
	virtual void suicide(Address _a) override final
	{
		m_s.addBalance(_a, m_s.balance(myAddress));
		m_s.subBalance(myAddress, m_s.balance(myAddress));
		ExtVMFace::suicide(_a);
	}

	/// Revert any changes made (by any of the other calls).
	/// @TODO check call site for the parent manifest being discarded.
	virtual void revert() override final
	{
		m_s.m_cache = m_origCache;
		sub.clear();
	}

	State& state() const { return m_s; }

private:
	State& m_s;											///< A reference to the base state.
	std::unordered_map<Address, Account> m_origCache;	///< The cache of the address states (i.e. the externalities) as-was prior to the execution.
};

}
}

