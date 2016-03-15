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
/** @file Common.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include <boost/algorithm/string/case_conv.hpp>
#include <libdevcore/Base64.h>
#include <libdevcore/Terminal.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include "Exceptions.h"
#include "Params.h"
#include "BlockInfo.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace eth
{

const unsigned c_protocolVersion = 61;
const unsigned c_minorProtocolVersion = 2;
const unsigned c_databaseBaseVersion = 9;
const unsigned c_databaseVersionModifier = 0;

const unsigned c_databaseVersion = c_databaseBaseVersion + (c_databaseVersionModifier << 8) + (23 << 9);

Address toAddress(std::string const& _s)
{
	try
	{
		auto b = fromHex(_s.substr(0, 2) == "0x" ? _s.substr(2) : _s, WhenError::Throw);
		if (b.size() == 20)
			return Address(b);
	}
	catch (BadHexCharacter&) {}
	BOOST_THROW_EXCEPTION(InvalidAddress());
}


vector<pair<u256, string>> const& units()
{
	static const vector<pair<u256, string>> s_units =
	{
		{exp10<54>(), "Uether"},
		{exp10<51>(), "Vether"},
		{exp10<48>(), "Dether"},
		{exp10<45>(), "Nether"},
		{exp10<42>(), "Yether"},
		{exp10<39>(), "Zether"},
		{exp10<36>(), "Eether"},
		{exp10<33>(), "Pether"},
		{exp10<30>(), "Tether"},
		{exp10<27>(), "Gether"},
		{exp10<24>(), "Mether"},
		{exp10<21>(), "grand"},
		{exp10<18>(), "ether"},
		{exp10<15>(), "finney"},
		{exp10<12>(), "szabo"},
		{exp10<9>(), "Gwei"},
		{exp10<6>(), "Mwei"},
		{exp10<3>(), "Kwei"},
		{exp10<0>(), "wei"}
	};

	return s_units;
}

std::string formatBalance(bigint const& _b)
{
	ostringstream ret;
	u256 b;
	if (_b < 0)
	{
		ret << "-";
		b = (u256)-_b;
	}
	else
		b = (u256)_b;

	if (b > units()[0].first * 1000)
	{
		ret << (b / units()[0].first) << " " << units()[0].second;
		return ret.str();
	}
	ret << setprecision(5);
	for (auto const& i: units())
		if (i.first != 1 && b >= i.first * 1)
		{
			ret << (double(b / (i.first / 1000)) / 1000.0) << " " << i.second;
			return ret.str();
		}
	ret << b << " wei";
	return ret.str();
}

static void badBlockInfo(BlockInfo const& _bi, string const& _err)
{
	string const c_line = EthReset EthOnMaroon + string(80, ' ') + EthReset;
	string const c_border = EthReset EthOnMaroon + string(2, ' ') + EthReset EthMaroonBold;
	string const c_space = c_border + string(76, ' ') + c_border + EthReset;
	stringstream ss;
	ss << c_line << endl;
	ss << c_space << endl;
	ss << c_border + "  Import Failure     " + _err + string(max<int>(0, 53 - _err.size()), ' ') + "  " + c_border << endl;
	ss << c_space << endl;
	string bin = toString(_bi.number());
	ss << c_border + ("                     Guru Meditation #" + string(max<int>(0, 8 - bin.size()), '0') + bin + "." + _bi.hash().abridged() + "                    ") + c_border << endl;
	ss << c_space << endl;
	ss << c_line;
	cwarn << "\n" + ss.str();
}

void badBlock(bytesConstRef _block, string const& _err)
{
	BlockInfo bi;
	DEV_IGNORE_EXCEPTIONS(bi = BlockInfo(_block, CheckNothing));
	badBlockInfo(bi, _err);
}

}
}
