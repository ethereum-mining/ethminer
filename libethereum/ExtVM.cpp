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
/** @file ExtVM.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "ExtVM.h"

#include "Executive.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

bool ExtVM::call(Address _receiveAddress, u256 _txValue, bytesConstRef _txData, u256& io_gas, bytesRef _out, OnOpFunc const& _onOp, Address _myAddressOverride, Address _codeAddressOverride)
{
	Executive e(m_s, lastHashes, depth + 1);
	if (!e.call(_receiveAddress, _codeAddressOverride ? _codeAddressOverride : _receiveAddress, _myAddressOverride ? _myAddressOverride : myAddress, _txValue, gasPrice, _txData, io_gas, origin))
	{
		e.go(_onOp);
		e.accrueSubState(sub);
	}
	io_gas = e.endGas();
	e.out().copyTo(_out);

	return !e.excepted();
}

h160 ExtVM::create(u256 _endowment, u256& io_gas, bytesConstRef _code, OnOpFunc const& _onOp)
{
	// Increment associated nonce for sender.
	m_s.noteSending(myAddress);

	Executive e(m_s, lastHashes, depth + 1);
	if (!e.create(myAddress, _endowment, gasPrice, io_gas, _code, origin))
	{
		e.go(_onOp);
		e.accrueSubState(sub);
	}
	io_gas = e.endGas();
	return e.newAddress();
}

