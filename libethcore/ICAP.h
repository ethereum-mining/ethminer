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
/** @file ICAP.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Ethereum-specific data structures & algorithms.
 */

#pragma once

#include <string>
#include <functional>
#include <libdevcore/Common.h>
#include <libdevcore/Exceptions.h>
#include <libdevcore/FixedHash.h>
#include "Common.h"

namespace dev
{
namespace eth
{

struct InvalidICAP: virtual public dev::Exception {};

class ICAP
{
public:
	ICAP() = default;
	ICAP(Address const& _a): m_direct(_a) {}
	ICAP(std::string const& _target): m_client(_target), m_asset("ETH") {}
	ICAP(std::string const& _client, std::string const& _inst): m_client(_client), m_institution(_inst), m_asset("XET") {}
	ICAP(std::string const& _c, std::string const& _i, std::string const& _a): m_client(_c), m_institution(_i), m_asset(_a) {}

	enum Type
	{
		Invalid,
		Direct,
		Indirect
	};

	static std::string iban(std::string _c, std::string _d);
	static std::pair<std::string, std::string> fromIBAN(std::string _iban);

	static ICAP decoded(std::string const& _encoded);
	std::string encoded() const;
	Type type() const { return m_type; }

	Address const& direct() const { return m_direct; }
	Address lookup(std::function<bytes(Address, bytes)> const& _call, Address const& _reg) const;
	Address address(std::function<bytes(Address, bytes)> const& _call, Address const& _reg) const { return m_type == Direct ? direct() : lookup(_call, _reg); }

private:
	Type m_type = Invalid;
	Address m_direct;
	std::string m_client;
	std::string m_institution;
	std::string m_asset;
};


}
}
