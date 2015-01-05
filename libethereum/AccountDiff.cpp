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
/** @file AccountDiff.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "AccountDiff.h"

#include <libdevcore/CommonIO.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

AccountChange AccountDiff::changeType() const
{
	bool bn = (balance || nonce);
	bool sc = (!storage.empty() || code);
	return exist ? exist.from() ? AccountChange::Deletion : AccountChange::Creation : (bn && sc) ? AccountChange::All : bn ? AccountChange::Intrinsic: sc ? AccountChange::CodeStorage : AccountChange::None;
}

char const* dev::eth::lead(AccountChange _c)
{
	switch (_c)
	{
	case AccountChange::None: return "   ";
	case AccountChange::Creation: return "+++";
	case AccountChange::Deletion: return "XXX";
	case AccountChange::Intrinsic: return " * ";
	case AccountChange::CodeStorage: return "* *";
	case AccountChange::All: return "***";
	}
	assert(false);
	return "";
}

namespace dev {

std::ostream& operator<<(std::ostream& _out, dev::eth::AccountDiff const& _s)
{
	if (!_s.exist.to())
		return _out;

	if (_s.nonce)
	{
		_out << std::dec << "#" << _s.nonce.to() << " ";
		if (_s.nonce.from())
			_out << "(" << std::showpos << (((bigint)_s.nonce.to()) - ((bigint)_s.nonce.from())) << std::noshowpos << ") ";
	}
	if (_s.balance)
	{
		_out << std::dec << _s.balance.to() << " ";
		if (_s.balance.from())
			_out << "(" << std::showpos << (((bigint)_s.balance.to()) - ((bigint)_s.balance.from())) << std::noshowpos << ") ";
	}
	if (_s.code)
		_out << "$" << std::hex << nouppercase << _s.code.to() << " (" << _s.code.from() << ") ";
	for (pair<u256, Diff<u256>> const& i: _s.storage)
		if (!i.second.from())
			_out << endl << " +     " << (h256)i.first << ": " << std::hex << nouppercase << i.second.to();
		else if (!i.second.to())
			_out << endl << "XXX    " << (h256)i.first << " (" << std::hex << nouppercase << i.second.from() << ")";
		else
			_out << endl << " *     " << (h256)i.first << ": " << std::hex << nouppercase << i.second.to() << " (" << i.second.from() << ")";
	return _out;
}

std::ostream& operator<<(std::ostream& _out, dev::eth::StateDiff const& _s)
{
	_out << _s.accounts.size() << " accounts changed:" << endl;
	dev::eth::AccountDiff d;
	_out << d;
	for (auto const& i: _s.accounts)
		_out << lead(i.second.changeType()) << "  " << i.first << ": " << i.second << endl;
	return _out;
}

}
