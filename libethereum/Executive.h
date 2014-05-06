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
/** @file Executive.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "CommonEth.h"
#include "Transaction.h"

namespace eth
{

class VM;
class ExtVM;
class State;

class Executive
{
public:
	Executive(State& _s): m_s(_s) {}
	~Executive();

	void setup(bytesConstRef _transaction);
	void create(Address _txSender, u256 _endowment, u256 _gasPrice, u256 _gas, bytesConstRef _code, Address _originAddress);
	void call(Address _myAddress, Address _txSender, u256 _txValue, u256 _gasPrice, bytesConstRef _txData, u256 _gas, Address _originAddress);
	bool go(uint64_t _steps = (unsigned)-1);
	void finalize();
	u256 gasUsed() const;

	Transaction const& t() const { return m_t; }

	u256 gas() const;

	bytesConstRef out() const { return m_out; }
	h160 newAddress() const { return m_newAddress; }

	VM const& vm() const { return *m_vm; }
	State const& state() const { return m_s; }
	ExtVM const& ext() const { return *m_ext; }

private:
	State& m_s;
	ExtVM* m_ext = nullptr;	// TODO: make safe.
	VM* m_vm = nullptr;
	bytesConstRef m_out;
	Address m_newAddress;

	Transaction m_t;
	Address m_sender;
	u256 m_endGas;
};

}
