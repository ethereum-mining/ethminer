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
#include "libdevcore/CommonJS.h"
#include "libdevcore/Common.h"
#include "AppContext.h"
#include "TransactionListModel.h"
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
										Address address, bytes _data, u256 _nonce, Secret _secret) const
{
	return Transaction(_value, _gasPrice, _gas, address, _data, _nonce, _secret);
}

Transaction TransactionBuilder::getDefaultCreationTransaction(dev::bytes _code, KeyPair _sender, u256 _nonce) const
{
	u256 gasPrice = 10000000000000;
	u256 gas = 1000000;
	u256 amount = 100;
	return getCreationTransaction(amount, gasPrice, gas, _code, _nonce, _sender.secret());
}

Transaction TransactionBuilder::getDefaultBasicTransaction(Address _contractAddress, dev::bytes _data, KeyPair _sender, u256 _nonce) const
{
	u256 gasPrice = 10000000000000;
	u256 gas = 1000000;
	u256 amount = 100;
	return getBasicTransaction(amount, gasPrice, gas, _contractAddress, _data, _nonce, _sender.secret());
}

Transaction TransactionBuilder::getTransaction(Address _contractAddress, dev::bytes _data, KeyPair _sender, u256 _nonce, dev::mix::TransactionSettings _tr)
{
	return getBasicTransaction(_tr.value, _tr.gasPrice, _tr.gas, _contractAddress, _data, _nonce, _sender.secret());
}


