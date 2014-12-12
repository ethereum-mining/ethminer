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

#include <functional>
#include <libdevcore/Log.h>
#include <libevmcore/Instruction.h>
#include <libethcore/CommonEth.h>
#include <libevm/VMFace.h>
#include "Transaction.h"

namespace dev
{
namespace eth
{

class State;
class ExtVM;
struct Manifest;

struct VMTraceChannel: public LogChannel { static const char* name() { return "EVM"; } static const int verbosity = 11; };


class Executive
{
public:
	Executive(State& _s, unsigned _level): m_s(_s), m_depth(_level) {}
	~Executive() = default;
	Executive(Executive const&) = delete;
	void operator=(Executive) = delete;

	bool setup(bytesConstRef _transaction);
	bool create(Address _txSender, u256 _endowment, u256 _gasPrice, u256 _gas, bytesConstRef _code, Address _originAddress);
	bool call(Address _myAddress, Address _codeAddress, Address _txSender, u256 _txValue, u256 _gasPrice, bytesConstRef _txData, u256 _gas, Address _originAddress);
	bool go(OnOpFunc const& _onOp = OnOpFunc());
	void finalize(OnOpFunc const& _onOp = OnOpFunc());
	u256 gasUsed() const;

	static OnOpFunc simpleTrace();

	Transaction const& t() const { return m_t; }

	u256 endGas() const { return m_endGas; }

	bytesConstRef out() const { return m_out; }
	h160 newAddress() const { return m_newAddress; }
	LogEntries const& logs() const { return m_logs; }
	bool excepted() const { return m_excepted; }

	VMFace const& vm() const { return *m_vm; }
	ExtVM const& ext() const { return *m_ext; }
	State const& state() const { return m_s; }

private:
	State& m_s;
	std::shared_ptr<ExtVM> m_ext;
	std::unique_ptr<VMFace> m_vm;
	bytes m_precompiledOut;				///< Used for the output when there is no VM for a contract (i.e. precompiled).
	bytesConstRef m_out;				///< Holds the copyable output.
	Address m_newAddress;

	Transaction m_t;
	bool m_isCreation;
	bool m_excepted = false;
	unsigned m_depth = 0;
	Address m_sender;
	u256 m_endGas;

	LogEntries m_logs;
};

}
}
