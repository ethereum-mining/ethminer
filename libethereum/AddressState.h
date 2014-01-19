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
	Normal,
	Contract
};

class AddressState
{
public:
	AddressState(AddressType _type = AddressType::Normal): m_type(_type), m_balance(0), m_nonce(0) {}

	AddressType type() const { return m_type; }
	u256& balance() { return m_balance; }
	u256 const& balance() const { return m_balance; }
	u256& nonce() { return m_nonce; }
	u256 const& nonce() const { return m_nonce; }
	std::map<u256, u256>& memory() { assert(m_type == AddressType::Contract); return m_memory; }
	std::map<u256, u256> const& memory() const { assert(m_type == AddressType::Contract); return m_memory; }

private:
	AddressType m_type;
	u256 m_balance;
	u256 m_nonce;
	// TODO: std::hash<u256> and then move to unordered_map.
	// Will need to sort on hash construction.
	std::map<u256, u256> m_memory;
};

}


