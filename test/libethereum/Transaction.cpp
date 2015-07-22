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
/** @file Transaction.cpp
 * @author Dmitrii Khokhlov <winsvega@mail.ru>
 * @date 2015
 * Transaaction test functions.
 */

#include "test/TestHelper.h"
#include <libethcore/Exceptions.h>
#include <libevm/VMFace.h>

/*std::ostream& dev::eth::operator<<(std::ostream& _out, ExecutionResult const& _er)
{
	_out << "{" << _er.gasUsed << ", " << _er.newAddress << ", " << toHex(_er.output) << "}";
	return _out;
}

std::ostream& dev::eth::operator<<(std::ostream& _out, TransactionException const& _er)
{
	switch (_er)
	{
		case TransactionException::None: _out << "None"; break;
		case TransactionException::BadRLP: _out << "BadRLP"; break;
		case TransactionException::InvalidFormat: _out << "InvalidFormat"; break;
		case TransactionException::OutOfGasIntrinsic: _out << "OutOfGasIntrinsic"; break;
		case TransactionException::InvalidSignature: _out << "InvalidSignature"; break;
		case TransactionException::InvalidNonce: _out << "InvalidNonce"; break;
		case TransactionException::NotEnoughCash: _out << "NotEnoughCash"; break;
		case TransactionException::OutOfGasBase: _out << "OutOfGasBase"; break;
		case TransactionException::BlockGasLimitReached: _out << "BlockGasLimitReached"; break;
		case TransactionException::BadInstruction: _out << "BadInstruction"; break;
		case TransactionException::BadJumpDestination: _out << "BadJumpDestination"; break;
		case TransactionException::OutOfGas: _out << "OutOfGas"; break;
		case TransactionException::OutOfStack: _out << "OutOfStack"; break;
		case TransactionException::StackUnderflow: _out << "StackUnderflow"; break;
		default: _out << "Unknown"; break;
	}
	return _out;
}

Transaction::Transaction(bytesConstRef _rlpData, CheckTransaction _checkSig):
	TransactionBase(_rlpData, _checkSig)
{
	if (_checkSig >= CheckTransaction::Cheap && !checkPayment())
		BOOST_THROW_EXCEPTION(OutOfGasIntrinsic() << RequirementError(gasRequired(), (bigint)gas()));
}

bigint Transaction::gasRequired() const
{
	if (!m_gasRequired)
		m_gasRequired = Transaction::gasRequired(m_data);
	return m_gasRequired;
}
*/

using namespace dev;
using namespace eth;

BOOST_AUTO_TEST_SUITE(libethereum)

BOOST_AUTO_TEST_CASE(toTransactionExceptionConvert)
{
	RLPException rlpEx("exception");//toTransactionException(*(dynamic_cast<Exception*>
	BOOST_CHECK_MESSAGE(toTransactionException(rlpEx) == TransactionException::BadRLP, "RLPException !=> TransactionException");
	OutOfGasIntrinsic oogEx;
	BOOST_CHECK_MESSAGE(toTransactionException(oogEx) == TransactionException::OutOfGasIntrinsic, "OutOfGasIntrinsic !=> TransactionException");
	InvalidSignature sigEx;
	BOOST_CHECK_MESSAGE(toTransactionException(sigEx) == TransactionException::InvalidSignature, "InvalidSignature !=> TransactionException");
	OutOfGasBase oogbEx;
	BOOST_CHECK_MESSAGE(toTransactionException(oogbEx) == TransactionException::OutOfGasBase, "OutOfGasBase !=> TransactionException");
	InvalidNonce nonceEx;
	BOOST_CHECK_MESSAGE(toTransactionException(nonceEx) == TransactionException::InvalidNonce, "InvalidNonce !=> TransactionException");
	NotEnoughCash cashEx;
	BOOST_CHECK_MESSAGE(toTransactionException(cashEx) == TransactionException::NotEnoughCash, "NotEnoughCash !=> TransactionException");
	BlockGasLimitReached blGasEx;
	BOOST_CHECK_MESSAGE(toTransactionException(blGasEx) == TransactionException::BlockGasLimitReached, "BlockGasLimitReached !=> TransactionException");
	BadInstruction badInsEx;
	BOOST_CHECK_MESSAGE(toTransactionException(badInsEx) == TransactionException::BadInstruction, "BadInstruction !=> TransactionException");
	BadJumpDestination badJumpEx;
	BOOST_CHECK_MESSAGE(toTransactionException(badJumpEx) == TransactionException::BadJumpDestination, "BadJumpDestination !=> TransactionException");
	OutOfGas oogEx2;
	BOOST_CHECK_MESSAGE(toTransactionException(oogEx2) == TransactionException::OutOfGas, "OutOfGas !=> TransactionException");
	OutOfStack oosEx;
	BOOST_CHECK_MESSAGE(toTransactionException(oosEx) == TransactionException::OutOfStack, "OutOfStack !=> TransactionException");
	StackUnderflow stackEx;
	BOOST_CHECK_MESSAGE(toTransactionException(stackEx) == TransactionException::StackUnderflow, "StackUnderflow !=> TransactionException");
	Exception notEx;
	BOOST_CHECK_MESSAGE(toTransactionException(notEx) == TransactionException::Unknown, "Unexpected should be TransactionException::Unknown");
}

BOOST_AUTO_TEST_SUITE_END()
