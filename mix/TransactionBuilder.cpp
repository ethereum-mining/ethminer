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
/** @file TransactionBuilder.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include "libethereum/Executive.h"
#include "libdevcore/Common.h"
#include "ApplicationCtx.h"
#include "TransactionBuilder.h"
using namespace dev::mix;
using namespace dev::eth;
using namespace dev;

Transaction TransactionBuilder::getCreationTransaction(u256 _value, u256 _gasPrice, u256 _gas,
										bytes _data, u256 _nonce, Secret _secret) const
{
	return Transaction(_value, _gasPrice, _gas, _data, _nonce, _secret);
}

Transaction TransactionBuilder::getBasicTransaction(u256 _value, u256 _gasPrice, u256 _gas,
										QString address, bytes _data, u256 _nonce, Secret _secret) const
{
	return Transaction(_value, _gasPrice, _gas, fromString(address), _data, _nonce, _secret);
}

int TransactionBuilder::fromHex(char _i) const
{
	if (_i >= '0' && _i <= '9')
		return _i - '0';
	if (_i >= 'a' && _i <= 'f')
		return _i - 'a' + 10;
	if (_i >= 'A' && _i <= 'F')
		return _i - 'A' + 10;
	BOOST_THROW_EXCEPTION(BadHexCharacter() << errinfo_invalidSymbol(_i));
}

bytes TransactionBuilder::fromHex(std::string const& _s) const
{
	unsigned s = (_s[0] == '0' && _s[1] == 'x') ? 2 : 0;
	std::vector<uint8_t> ret;
	ret.reserve((_s.size() - s + 1) / 2);

	if (_s.size() % 2)
		try
		{
			ret.push_back(fromHex(_s[s++]));
		}
		catch (...){ ret.push_back(0); cwarn << boost::current_exception_diagnostic_information(); }
	for (unsigned i = s; i < _s.size(); i += 2)
		try
		{
			ret.push_back((byte)(fromHex(_s[i]) * 16 + fromHex(_s[i + 1])));
		}
		catch (...){ ret.push_back(0); cwarn << boost::current_exception_diagnostic_information(); }
	return ret;
}

Address TransactionBuilder::fromString(QString const& _a) const
{
	Client* ethClient = ApplicationCtx::getInstance()->getEthereumClient();
	std::string sn = _a.toStdString();
	if (sn.size() > 32)
		sn.resize(32);
	h256 n;
	memcpy(n.data(), sn.data(), sn.size());
	memset(n.data() + sn.size(), 0, 32 - sn.size());
	if (_a.size() == 40)
		return Address(fromHex(_a.toStdString()));
	else
	{
		//we try to resolve the recipient adress using nameReg contract state
		const Address c_config = Address("661005d2720d855f1d9976f88bb10c1a3398c77f"); //NameReg contract
		if (h160 nameReg = (u160)ethClient->stateAt(c_config, 0))
			if (h256 a = ethClient->stateAt(nameReg, n))
				return right160(a);
	}
	return Address(); // should maybe throws exception instead of returning blank address.
}
