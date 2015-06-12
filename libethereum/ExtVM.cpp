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
#include <exception>
#include <boost/thread.hpp>
#include "Executive.h"

using namespace dev;
using namespace dev::eth;

namespace
{
static unsigned const c_depthLimit = 1024;
static size_t const c_singleExecutionStackSize = 12 * 1024;
static size_t const c_defaultStackSize = 512 * 1024;
static unsigned const c_offloadPoint = c_defaultStackSize / c_singleExecutionStackSize;

void goOnOffloadedStack(Executive& _e, OnOpFunc const& _onOp)
{
	cnote << "CALL OFFLOADING: offloading point " << c_offloadPoint;
	boost::thread::attributes attrs;
	attrs.set_stack_size((c_depthLimit - c_offloadPoint) * c_singleExecutionStackSize);

	std::exception_ptr exception;
	boost::thread{attrs, [&]{
		cnote << "OFFLOADING thread";
		try
		{
			_e.go(_onOp);
		}
		catch (...)
		{
			cnote << "!!!!!!!!!!! exception in offloading!!!!!!!!!!!!";
			exception = std::current_exception();
		}
	}}.join();
	if (exception)
		std::rethrow_exception(exception);
}

void go(unsigned _depth, Executive& _e, OnOpFunc const& _onOp)
{
	if (_depth == c_offloadPoint)
		goOnOffloadedStack(_e, _onOp);
	else
		_e.go(_onOp);
}
}

bool ExtVM::call(CallParameters& _p)
{
	Executive e(m_s, lastHashes, depth + 1);
	if (!e.call(_p, gasPrice, origin))
	{
		go(depth, e, _p.onOp);
		e.accrueSubState(sub);
	}
	_p.gas = e.gas();

	return !e.excepted();
}

h160 ExtVM::create(u256 _endowment, u256& io_gas, bytesConstRef _code, OnOpFunc const& _onOp)
{
	// Increment associated nonce for sender.
	m_s.noteSending(myAddress);

	Executive e(m_s, lastHashes, depth + 1);
	if (!e.create(myAddress, _endowment, gasPrice, io_gas, _code, origin))
	{
		go(depth, e, _onOp);
		e.accrueSubState(sub);
	}
	io_gas = e.gas();
	return e.newAddress();
}

