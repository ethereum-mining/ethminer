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
class BlockChain;
class ExtVM;
struct Manifest;

struct VMTraceChannel: public LogChannel { static const char* name() { return "EVM"; } static const int verbosity = 11; };

/**
 * @brief Message-call/contract-creation executor; useful for executing transactions.
 *
 * Two ways of using this class - either as a transaction executive or a CALL/CREATE executive.
 * In the first use, after construction, begin with setup() and end with finalize(). Call go()
 * after setup() only if it returns false.
 * In the second use, after construction, begin with call() or create() and end with
 * accrueSubState(). Call go() after call()/create() only if it returns false.
 */
class Executive
{
public:
	/// Basic constructor.
	Executive(State& _s, LastHashes const& _lh, unsigned _level): m_s(_s), m_lastHashes(_lh), m_depth(_level) {}
	/// Basic constructor.
	Executive(State& _s, BlockChain const& _bc, unsigned _level);
	/// Basic destructor.
	~Executive() = default;

	Executive(Executive const&) = delete;
	void operator=(Executive) = delete;

	/// Set up the executive for evaluating a transaction. You must call finalize() following this.
	/// @returns true iff go() must be called (and thus a VM execution in required).
	bool setup(bytesConstRef _transaction);
	/// Finalise a transaction previously set up with setup().
	/// @warning Only valid after setup(), and possibly go().
	void finalize();
	/// @returns the transaction from setup().
	/// @warning Only valid after setup().
	Transaction const& t() const { return m_t; }
	/// @returns the log entries created by this operation.
	/// @warning Only valid after finalise().
	LogEntries const& logs() const { return m_logs; }
	/// @returns total gas used in the transaction/operation.
	/// @warning Only valid after finalise().
	u256 gasUsed() const;

	/// Set up the executive for evaluating a bare CREATE (contract-creation) operation.
	/// @returns false iff go() must be called (and thus a VM execution in required).
	bool create(Address _txSender, u256 _endowment, u256 _gasPrice, u256 _gas, bytesConstRef _code, Address _originAddress);
	/// Set up the executive for evaluating a bare CALL (message call) operation.
	/// @returns false iff go() must be called (and thus a VM execution in required).
	bool call(Address _myAddress, Address _codeAddress, Address _txSender, u256 _txValue, u256 _gasPrice, bytesConstRef _txData, u256 _gas, Address _originAddress);
	/// Finalise an operation through accruing the substate into the parent context.
	void accrueSubState(SubState& _parentContext);

	/// Executes (or continues execution of) the VM.
	/// @returns false iff go() must be called again to finish the transction.
	bool go(OnOpFunc const& _onOp = OnOpFunc());

	/// Operation function for providing a simple trace of the VM execution.
	static OnOpFunc simpleTrace();

	/// @returns gas remaining after the transaction/operation.
	u256 endGas() const { return m_endGas; }
	/// @returns output data of the transaction/operation.
	bytesConstRef out() const { return m_out; }
	/// @returns the new address for the created contract in the CREATE operation.
	h160 newAddress() const { return m_newAddress; }
	/// @returns true iff the operation ended with a VM exception.
	bool excepted() const { return m_excepted; }

private:
	State& m_s;							///< The state to which this operation/transaction is applied.
	LastHashes m_lastHashes;
	std::shared_ptr<ExtVM> m_ext;		///< The VM externality object for the VM execution or null if no VM is required.
	std::unique_ptr<VMFace> m_vm;		///< The VM object or null if no VM is required.
	bytes m_precompiledOut;				///< Used for the output when there is no VM for a contract (i.e. precompiled).
	bytesConstRef m_out;				///< The copyable output.
	Address m_newAddress;				///< The address of the created contract in the case of create() being called.

	unsigned m_depth = 0;				///< The context's call-depth.
	bool m_isCreation = false;			///< True if the transaction creates a contract, or if create() is called.
	bool m_excepted = false;			///< True if the VM execution resulted in an exception.
	u256 m_endGas;						///< The final amount of gas for the transaction.

	Transaction m_t;					///< The original transaction. Set by setup().
	LogEntries m_logs;					///< The log entries created by this transaction. Set by finalize().
};

}
}
