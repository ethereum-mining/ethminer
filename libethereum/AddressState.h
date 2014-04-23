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
/** @file AddressState.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libethcore/Common.h>
#include <libethcore/RLP.h>

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
	AddressState(): m_type(AddressType::Dead), m_balance(0), m_nonce(0), m_isComplete(false) {}
	AddressState(u256 _balance, u256 _nonce, AddressType _type = AddressType::Normal): m_type(_type), m_balance(_balance), m_nonce(_nonce), m_isComplete(true) {}
	AddressState(u256 _balance, u256 _nonce, h256 _contractRoot, h256 _codeHash): m_type(AddressType::Contract), m_balance(_balance), m_nonce(_nonce), m_isComplete(false), m_contractRoot(_contractRoot), m_codeHash(_codeHash) {}
	AddressState(u256 _balance, u256 _nonce, bytesConstRef _code);

	void incNonce() { m_nonce++; }
	void addBalance(bigint _i) { m_balance = (u256)((bigint)m_balance + _i); }
	void kill() { m_type = AddressType::Dead; m_memory.clear(); m_contractRoot = h256(); m_balance = 0; m_nonce = 0; }

	AddressType type() const { return m_type; }
	u256& balance() { return m_balance; }
	u256 const& balance() const { return m_balance; }
	u256& nonce() { return m_nonce; }
	u256 const& nonce() const { return m_nonce; }
	bool isComplete() const { return m_isComplete; }
	std::map<u256, u256>& setIsComplete(bytesConstRef _code) { assert(m_type == AddressType::Contract); m_isComplete = true; m_contractRoot = h256(); m_code = _code.toBytes(); return m_memory; }
	h256 oldRoot() const { assert(!isComplete()); return m_contractRoot; }
	h256 codeHash() const { assert(m_codeHash); return m_codeHash; }
	std::map<u256, u256>& memory() { assert(m_type == AddressType::Contract && isComplete()); return m_memory; }
	std::map<u256, u256> const& memory() const { assert(m_type == AddressType::Contract && isComplete()); return m_memory; }
	bytes const& code() const { assert(m_type == AddressType::Contract && isComplete()); return m_code; }
	bool freshCode() const { return !m_codeHash && m_isComplete; }

private:
	AddressType m_type;
	u256 m_balance;
	u256 m_nonce;
	bool m_isComplete;
	h256 m_contractRoot;
	h256 m_codeHash;	// if 0 and m_isComplete, has been created and needs to be inserted.
	// TODO: change to unordered_map.
	std::map<u256, u256> m_memory;
	bytes m_code;
};

}


