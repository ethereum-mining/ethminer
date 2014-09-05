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
/** @file AccountDiff.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libethential/Common.h>
#include <libethcore/CommonEth.h>

namespace dev
{
namespace eth
{

enum class ExistDiff { Same, New, Dead };
template <class T>
class Diff
{
public:
	Diff() {}
	Diff(T _from, T _to): m_from(_from), m_to(_to) {}

	T const& from() const { return m_from; }
	T const& to() const { return m_to; }

	explicit operator bool() const { return m_from != m_to; }

private:
	T m_from;
	T m_to;
};

enum class AccountChange { None, Creation, Deletion, Intrinsic, CodeStorage, All };

struct AccountDiff
{
	inline bool changed() const { return storage.size() || code || nonce || balance || exist; }
	char const* lead() const;
	AccountChange changeType() const;

	Diff<bool> exist;
	Diff<u256> balance;
	Diff<u256> nonce;
	std::map<u256, Diff<u256>> storage;
	Diff<bytes> code;
};

struct StateDiff
{
	std::map<Address, AccountDiff> accounts;
};

}

std::ostream& operator<<(std::ostream& _out, dev::eth::StateDiff const& _s);
std::ostream& operator<<(std::ostream& _out, dev::eth::AccountDiff const& _s);

}

