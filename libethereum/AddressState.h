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

// TODO: Document fully.

class AddressState
{
public:
	AddressState(): m_isAlive(false), m_balance(0), m_nonce(0) {}
	AddressState(u256 _balance, u256 _nonce, h256 _contractRoot, h256 _codeHash): m_isAlive(true), m_balance(_balance), m_nonce(_nonce), m_storageRoot(_contractRoot), m_codeHash(_codeHash) {}

	void kill() { m_isAlive = false; m_storageOverlay.clear(); m_codeHash = EmptySHA3; m_storageRoot = h256(); m_balance = 0; m_nonce = 0; }
	bool isAlive() const { return m_isAlive; }

	u256& balance() { return m_balance; }
	u256 const& balance() const { return m_balance; }
	void addBalance(bigint _i) { m_balance = (u256)((bigint)m_balance + _i); }

	u256& nonce() { return m_nonce; }
	u256 const& nonce() const { return m_nonce; }
	void incNonce() { m_nonce++; }

	h256 oldRoot() const { return m_storageRoot; }
	std::map<u256, u256> const& storage() const { return m_storageOverlay; }
	void setStorage(u256 _p, u256 _v) { m_storageOverlay[_p] = _v; }

	bool isFreshCode() const { return !m_codeHash; }
	bool codeBearing() const { return m_codeHash != EmptySHA3; }
	bool codeCacheValid() const { return m_codeHash == EmptySHA3 || !m_codeHash || m_codeCache.size(); }
	h256 codeHash() const { assert(m_codeHash); return m_codeHash; }
	bytes const& code() const { assert(m_codeHash == EmptySHA3 || !m_codeHash || m_codeCache.size()); return m_codeCache; }
	void setCode(bytesConstRef _code) { assert(!m_codeHash); m_codeCache = _code.toBytes(); }
	void noteCode(bytesConstRef _code) { assert(sha3(_code) == m_codeHash); m_codeCache = _code.toBytes(); }

private:
	bool m_isAlive;
	bool m_gotCode;
	u256 m_balance;
	u256 m_nonce;

	/// The base storage root. Used with the state DB to give a base to the storage. m_storageOverlay is overlaid on this and takes precedence for all values set.
	h256 m_storageRoot;

	/// If 0 then we're in the limbo where we're running the initialisation code. We expect a setCode() at some point later.
	/// If EmptySHA3, then m_code, which should be empty, is valid.
	/// If anything else, then m_code is valid iff it's not empty, otherwise, State::ensureCached() needs to be called with the correct args.
	h256 m_codeHash;

	// TODO: change to unordered_map.
	std::map<u256, u256> m_storageOverlay;
	bytes m_codeCache;
};

}


