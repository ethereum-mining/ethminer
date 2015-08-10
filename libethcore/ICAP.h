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
#include <boost/algorithm/string/case_conv.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/Exceptions.h>
#include <libdevcore/FixedHash.h>
#include "Common.h"

namespace dev
{
namespace eth
{

DEV_SIMPLE_EXCEPTION(InvalidICAP);

/**
 * @brief Encapsulation of an ICAP address.
 * Can be encoded, decoded, looked-up and inspected.
 */
class ICAP
{
public:
	/// Construct null ICAP object.
	ICAP() = default;
	/// Construct a direct ICAP object for given target address. Must have a zero first byte.
	ICAP(Address const& _target): m_type(Direct), m_direct(_target) {}
	/// Construct an indirect ICAP object for given client and institution names.
	ICAP(std::string const& _client, std::string const& _inst): m_type(Indirect), m_client(boost::algorithm::to_upper_copy(_client)), m_institution(boost::algorithm::to_upper_copy(_inst)), m_asset("XET") {}
	/// Construct an indirect ICAP object for given client, institution and asset names. You generally don't want to use this.
	ICAP(std::string const& _c, std::string const& _i, std::string const& _a): m_type(Indirect), m_client(boost::algorithm::to_upper_copy(_c)), m_institution(boost::algorithm::to_upper_copy(_i)), m_asset(boost::algorithm::to_upper_copy(_a)) {}

	/// Type of ICAP address.
	enum Type
	{
		Invalid,
		Direct,
		Indirect
	};

	/// Create a direct address for ICAP.
	static Secret createDirect();

	/// @returns IBAN encoding of client and data.
	static std::string iban(std::string _c, std::string _d);
	/// @returns Client and data from given IBAN address.
	static std::pair<std::string, std::string> fromIBAN(std::string _iban);

	/// @returns the ICAP object for the ICAP address given.
	static ICAP decoded(std::string const& _encoded);

	/// @returns the encoded ICAP address.
	std::string encoded() const;
	/// @returns type of ICAP.
	Type type() const { return m_type; }
	/// @returns target address. Only valid when type() == Direct.
	Address const& direct() const { return m_type == Direct ? m_direct : ZeroAddress; }
	/// @returns asset. Only valid when type() == Indirect.
	std::string const& asset() const { return m_type == Indirect ? m_asset : EmptyString; }
	/// @returns target name. Only valid when type() == Indirect and asset() == "ETH".
	std::string const& target() const { return m_type == Indirect && m_asset == "ETH" ? m_client : EmptyString; }
	/// @returns institution name. Only valid when type() == Indirect and asset() == "XET".
	std::string const& institution() const { return m_type == Indirect && m_asset == "XET" ? m_institution : EmptyString; }
	/// @returns client name. Only valid when type() == Indirect and asset() == "XET".
	std::string const& client() const { return m_type == Indirect && m_asset == "XET" ? m_client : EmptyString; }
	/// @returns target address. Always valid, but requires the Registry address and a function to make calls.
	std::pair<Address, bytes> address(std::function<bytes(Address, bytes)> const& _call, Address const& _reg) const { return m_type == Direct ? make_pair(direct(), bytes()) : m_type == Indirect ? lookup(_call, _reg) : make_pair(Address(), bytes()); }

	/// @returns target address. Looks up through the given Registry and call function. Only valid when type() == Indirect.
	std::pair<Address, bytes> lookup(std::function<bytes(Address, bytes)> const& _call, Address const& _reg) const;

private:
	Type m_type = Invalid;
	Address m_direct;
	std::string m_client;
	std::string m_institution;
	std::string m_asset;
};


}
}
