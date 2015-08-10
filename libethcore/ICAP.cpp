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
/** @file ICAP.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "ICAP.h"
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string.hpp>
#include <libdevcore/Base64.h>
#include <libdevcore/SHA3.h>
#include "Exceptions.h"
#include "ABI.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace eth
{

string ICAP::iban(std::string _c, std::string _d)
{
	boost::to_upper(_c);
	boost::to_upper(_d);
	auto totStr = _d + _c + "00";
	bigint tot = 0;
	for (char x: totStr)
		if (x >= 'A')
			tot = tot * 100 + x - 'A' + 10;
		else
			tot = tot * 10 + x - '0';
	unsigned check = (unsigned)(u256)(98 - tot % 97);
	ostringstream out;
	out << _c << setfill('0') << setw(2) << check << _d;
	return out.str();
}

std::pair<string, string> ICAP::fromIBAN(std::string _iban)
{
	if (_iban.size() < 4)
		return std::make_pair(string(), string());
	boost::to_upper(_iban);
	std::string c = _iban.substr(0, 2);
	std::string d = _iban.substr(4);
	if (iban(c, d) != _iban)
		return std::make_pair(string(), string());
	return make_pair(c, d);
}

Secret ICAP::createDirect()
{
	Secret ret;
	while (true)
	{
		ret = Secret::random();
		if (!toAddress(ret)[0])
			return ret;
	}
}

ICAP ICAP::decoded(std::string const& _encoded)
{
	ICAP ret;
	std::string country;
	std::string data;
	std::tie(country, data) = fromIBAN(_encoded);
	if (country != "XE")
		BOOST_THROW_EXCEPTION(InvalidICAP());
	if (data.size() == 30 || data.size() == 31)
	{
		ret.m_type = Direct;
		// Direct ICAP
		ret.m_direct = fromBase36<Address::size>(data);
	}
	else if (data.size() == 16)
	{
		ret.m_type = Indirect;
		ret.m_asset = data.substr(0, 3);
		if (ret.m_asset == "XET" || ret.m_asset == "ETH")
		{
			ret.m_institution = data.substr(3, 4);
			ret.m_client = data.substr(7);
		}
		else
			BOOST_THROW_EXCEPTION(InvalidICAP());
	}
	else
		BOOST_THROW_EXCEPTION(InvalidICAP());

	return ret;
}

std::string ICAP::encoded() const
{
	if (m_type == Direct)
	{
		std::string d = toBase36<Address::size>(m_direct);
		while (d.size() < 30)		// always 34, sometimes 35.
			d = "0" + d;
		return iban("XE", d);
	}
	else if (m_type == Indirect)
	{
		if (
			m_asset.find_first_not_of("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890") != string::npos ||
			m_institution.find_first_not_of("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890") != string::npos ||
			m_client.find_first_not_of("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890") != string::npos ||
			m_asset.size() != 3 ||
			(boost::to_upper_copy(m_asset) != "XET" && boost::to_upper_copy(m_asset) != "ETH") ||
			m_institution.size() != 4 ||
			m_client.size() != 9
		)
			BOOST_THROW_EXCEPTION(InvalidICAP());
		return iban("XE", m_asset + m_institution + m_client);
	}
	else
		BOOST_THROW_EXCEPTION(InvalidICAP());
}

pair<Address, bytes> ICAP::lookup(std::function<bytes(Address, bytes)> const& _call, Address const& _reg) const
{
	auto resolve = [&](string const& s)
	{
		vector<string> ss;
		boost::algorithm::split(ss, s, boost::is_any_of("/"));
		Address r = _reg;
		for (unsigned i = 0; i < ss.size() - 1; ++i)
			r = abiOut<Address>(_call(r, abiIn("subRegistrar(bytes32)", toString32(ss[i]))));
		return abiOut<Address>(_call(r, abiIn("addr(bytes32)", toString32(ss.back()))));
	};
	if (m_asset == "XET")
	{
		Address a = resolve(m_institution);
		bytes d = abiIn("deposit(uint64)", fromBase36<8>(m_client));
		return make_pair(a, d);
	}
	else if (m_asset == "ETH")
	{
		if (m_institution == "XREG")
			return make_pair(resolve(m_client), bytes());
		else if (m_institution[0] != 'X')
			return make_pair(resolve(m_institution + "/" + m_client), bytes());
		else
			BOOST_THROW_EXCEPTION(InterfaceNotSupported("ICAP::lookup(), bad institution"));
	}
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("ICAP::lookup(), bad asset"));
}

}
}
