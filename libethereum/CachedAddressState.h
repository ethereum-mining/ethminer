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
/** @file CachedAddressState.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <string>
#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include "AccountDiff.h"

namespace dev
{

class OverlayDB;

namespace eth
{

class Account;

class CachedAddressState
{
public:
	CachedAddressState(std::string const& _rlp, Account const* _s, OverlayDB const* _o): m_rS(_rlp), m_r(m_rS), m_s(_s), m_o(_o) {}

	bool exists() const;
	u256 balance() const;
	u256 nonce() const;
	bytes code() const;

	// TODO: DEPRECATE.
	std::map<u256, u256> storage() const;

	AccountDiff diff(CachedAddressState const& _c);

private:
	std::string m_rS;
	RLP m_r;
	Account const* m_s;
	OverlayDB const* m_o;
};

}

}
