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

/// Upper bound of stack space needed by single CALL/CREATE execution. Set experimentally.
static size_t const c_singleExecutionStackSize =
#ifdef NDEBUG
	12 * 1024;
#else
	33 * 1024;
#endif

/// Standard OSX thread stack limit. Should be reasonable for other platforms too.
static size_t const c_defaultStackSize = 512 * 1024;

/// Stack overhead prior to allocation.
static size_t const c_entryOverhead = 128 * 1024;

/// On what depth execution should be offloaded to additional separated stack space.
static unsigned const c_offloadPoint = (c_defaultStackSize - c_entryOverhead) / c_singleExecutionStackSize;

void goOnOffloadedStack(Executive& _e, OnOpFunc const& _onOp)
{
	// Set new stack size enouth to handle the rest of the calls up to the limit.
	boost::thread::attributes attrs;
	attrs.set_stack_size((c_depthLimit - c_offloadPoint) * c_singleExecutionStackSize);

	// Create new thread with big stack and join immediately.
	// TODO: It is possible to switch the implementation to Boost.Context or similar when the API is stable.
	std::exception_ptr exception;
	boost::thread{attrs, [&]{
		try
		{
			_e.go(_onOp);
		}
		catch (...)
		{
			exception = std::current_exception(); // Catch all exceptions to be rethrown in parent thread.
		}
	}}.join();
	if (exception)
		std::rethrow_exception(exception);
}

void go(unsigned _depth, Executive& _e, OnOpFunc const& _onOp)
{
	// If in the offloading point we need to switch to additional separated stack space.
	// Current stack is too small to handle more CALL/CREATE executions.
	// It needs to be done only once as newly allocated stack space it enough to handle
	// the rest of the calls up to the depth limit (c_depthLimit).

	if (_depth == c_offloadPoint)
		goOnOffloadedStack(_e, _onOp);
	else
		_e.go(_onOp);
}
}

bool ExtVM::call(CallParameters& _p)
{
	Executive e(m_s, envInfo(), depth + 1);
	if (!e.call(_p, gasPrice, origin))
	{
#if __clang__ // Enabled for clang only as the problem affects OSX
		go(depth, e, _p.onOp);
#else
		e.go(_p.onOp);
#endif
		e.accrueSubState(sub);
	}
	_p.gas = e.gas();

	return !e.excepted();
}

h160 ExtVM::create(u256 _endowment, u256& io_gas, bytesConstRef _code, OnOpFunc const& _onOp)
{
	// Increment associated nonce for sender.
	m_s.noteSending(myAddress);

	Executive e(m_s, envInfo(), depth + 1);
	if (!e.create(myAddress, _endowment, gasPrice, io_gas, _code, origin))
	{
		go(depth, e, _onOp);
		e.accrueSubState(sub);
	}
	io_gas = e.gas();
	return e.newAddress();
}
