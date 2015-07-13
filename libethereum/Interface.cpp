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
/** @file Interface.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Interface.h"
using namespace std;
using namespace dev;
using namespace eth;

void Interface::submitTransaction(Secret const& _secret, u256 const& _value, Address const& _dest, bytes const& _data, u256 const& _gas, u256 const& _gasPrice, u256 const& _nonce)
{
	TransactionSkeleton ts;
	ts.creation = false;
	ts.value = _value;
	ts.to = _dest;
	ts.data = _data;
	ts.gas = _gas;
	ts.gasPrice = _gasPrice;
	ts.nonce = _nonce;
	submitTransaction(ts, _secret);
}

Address Interface::submitTransaction(Secret const& _secret, u256 const& _endowment, bytes const& _init, u256 const& _gas, u256 const& _gasPrice, u256 const& _nonce)
{
	TransactionSkeleton ts;
	ts.creation = true;
	ts.value = _endowment;
	ts.data = _init;
	ts.gas = _gas;
	ts.gasPrice = _gasPrice;
	ts.nonce = _nonce;
	return submitTransaction(ts, _secret).second;
}
