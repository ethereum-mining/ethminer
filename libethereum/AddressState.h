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

// TODO: Don't pre-cache all of storage.

class AddressState
{
public:
	AddressState(): m_isAlive(false), m_isComplete(false), m_balance(0), m_nonce(0) {}
	AddressState(u256 _balance, u256 _nonce, h256 _contractRoot, h256 _codeHash): m_isAlive(true), m_isComplete(_codeHash == EmptySHA3 && !_contractRoot), m_balance(_balance), m_nonce(_nonce), m_storageRoot(_contractRoot), m_codeHash(_codeHash) {}
	AddressState(u256 _balance, u256 _nonce, bytesConstRef _code);

	void kill() { m_isAlive = false; m_storage.clear(); m_codeHash = EmptySHA3; m_storageRoot = h256(); m_balance = 0; m_nonce = 0; }
	bool isAlive() const { return m_isAlive; }

	u256& balance() { return m_balance; }
	u256 const& balance() const { return m_balance; }
	void addBalance(bigint _i) { m_balance = (u256)((bigint)m_balance + _i); }

	u256& nonce() { return m_nonce; }
	u256 const& nonce() const { return m_nonce; }
	void incNonce() { m_nonce++; }

	bool isComplete() const { return m_isComplete; }
	std::map<u256, u256>& setIsComplete(bytesConstRef _code) { m_isComplete = true; m_storageRoot = h256(); m_code = _code.toBytes(); return m_storage; }

	h256 oldRoot() const { assert(!isComplete()); return m_storageRoot; }
	std::map<u256, u256>& memory() { return m_storage; }
	std::map<u256, u256> const& memory() const { assert(isComplete()); return m_storage; }

	h256 codeHash() const { assert(m_codeHash); return m_codeHash; }
	bytes const& code() const { assert(isComplete()); return m_code; }
	bool freshCode() const { return !m_codeHash && m_isComplete; }
	void setCode(bytesConstRef _code) { assert(freshCode()); m_code = _code.toBytes(); }

private:
	bool m_isAlive;
	bool m_isComplete;
	bool m_gotCode;
	u256 m_balance;
	u256 m_nonce;
	h256 m_storageRoot;
	h256 m_codeHash;	// if 0 and m_isComplete, has been created and needs to be inserted.
	// TODO: change to unordered_map.
	std::map<u256, u256> m_storage;
	bytes m_code;
};

}


