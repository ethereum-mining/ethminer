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
/** @file AddressState.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "Common.h"
#include "RLP.h"

namespace eth
{

enum class AddressType
{
	Dead,
	Normal,
	Contract
};

class AddressState
{
public:
	AddressState(): m_type(AddressType::Dead), m_balance(0), m_nonce(0) {}
	AddressState(u256 _balance, u256 _nonce): m_type(AddressType::Normal), m_balance(_balance), m_nonce(_nonce) {}
	AddressState(u256 _balance, u256 _nonce, h256 _contractRoot): m_type(AddressType::Contract), m_balance(_balance), m_nonce(_nonce), m_contractRoot(_contractRoot) {}

	void incNonce() { m_nonce++; }
	void addBalance(bigint _i) { m_balance = (u256)((bigint)m_balance + _i); }
	void kill() { m_type = AddressType::Dead; m_memory.clear(); m_contractRoot = h256(); m_balance = 0; m_nonce = 0; }

	AddressType type() const { return m_type; }
	u256& balance() { return m_balance; }
	u256 const& balance() const { return m_balance; }
	u256& nonce() { return m_nonce; }
	u256 const& nonce() const { return m_nonce; }
	bool haveMemory() const { return m_memory.empty() && m_contractRoot != h256(); }	// TODO: best to switch to m_haveMemory flag rather than try to infer.
	h256 oldRoot() const { assert(!haveMemory()); return m_contractRoot; }
	std::map<u256, u256>& takeMemory() { assert(m_type == AddressType::Contract && haveMemory()); m_contractRoot = h256(); return m_memory; }
	std::map<u256, u256> const& memory() const { assert(m_type == AddressType::Contract && haveMemory()); return m_memory; }

private:
	AddressType m_type;
	u256 m_balance;
	u256 m_nonce;
	h256 m_contractRoot;
	// TODO: change to unordered_map.
	std::map<u256, u256> m_memory;
};

}


